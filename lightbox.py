from octoai.client import Client
import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
import os


# These need to be set in your environment
SEGMENT_TOKEN = os.environ["SEGMENT_TOKEN"]
INPAINT_TOKEN = os.environ["INPAINT_TOKEN"]
SEGMENT_ENDPOINT = os.environ["SEGMENT_ENDPOINT"]
INPAINT_ENDPOINT = os.environ["INPAINT_ENDPOINT"]


def image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64


def crop_centered(image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size

    left = (original_width - target_width) / 2
    top = (original_height - target_height) / 2
    right = (original_width + target_width) / 2
    bottom = (original_height + target_height) / 2

    return image.crop((left, top, right, bottom))


def rescale_image(image):
    w, h = image.size
    if w == h:
        width = 1024
        height = 1024
    elif w > h:
        width = 1024
        height = 1024 * h // w
    else:
        width = 1024 * w // h
        height = 1024
    image = image.resize((width, height))
    return image


def generate_gallery(my_upload, meta_prompt, num_images=3):

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    input_img = Image.open(my_upload)
    client = Client(token=SEGMENT_TOKEN)
    inputs = {
        "input": {
            "image": image_to_base64(input_img)
        }
    }

    # perform inference
    response = client.infer(endpoint_url=f"{SEGMENT_ENDPOINT}/infer", inputs=inputs)

    # get the crop mask
    crop_mask_string = response["output"]["crop_mask"]
    crop_mask = Image.open(BytesIO(b64decode(crop_mask_string)))
    image = rescale_image(input_img)
    mask_image = rescale_image(crop_mask)

    # Get cropped
    im_rgb = image.convert("RGB")
    cropped = im_rgb.copy()
    cropped.putalpha(mask_image.convert('L'))

    # do inpainting
    inputs = {
        "input": {
            "prompt": meta_prompt,
            "negative_prompt": "ugly, distorted, low res",
            "image": image_to_base64(image),
            "mask_image": image_to_base64(mask_image),
            "mask_source": "MASK_IMAGE_BLACK",
            "guidance_scale": 7.5,
            "num_inference_steps": 20,
            "strength": 0.99,
            "num_images": 3,
            "sampler": "K_EULER_ANCESTRAL",
            "style_preset": "ads-advertising",
            # "use_refiner": True,
        }
    }

    client = Client(token=INPAINT_TOKEN)
    response = client.infer(endpoint_url=f"{INPAINT_ENDPOINT}/infer", inputs=inputs)
    print(
        "Generation time: {0:.2f} seconds".format(response["output"]["generation_time"])
    )
    for i, img in enumerate(response["output"]["images"]):
        image = Image.open(BytesIO(b64decode(img["base64"])))

        composite = Image.alpha_composite(
            image.convert('RGBA'),
            crop_centered(cropped, image.size)
        )
        cols[i%len(cols)].image(composite)

    print("Done generating the gallery!")

st.set_page_config(layout="wide", page_title="LightBox")

st.write("## LightBox - Powered by OctoAI")

meta_prompt = st.text_input("Product setting", value="forest, moss, mushrooms")
my_upload = st.file_uploader("Upload a product photo", type=["png", "jpg", "jpeg"])

if my_upload:
    generate_gallery(my_upload, meta_prompt)
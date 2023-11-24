from octoai.client import Client
import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
import os
import random
import time

# These need to be set in your environment
OCTOAI_TOKEN = os.environ["OCTOAI_TOKEN"]
CLIP_ENDPOINT = os.environ["CLIP_ENDPOINT"]
DEPTH_MASK_ENDPOINT = os.environ["DEPTH_MASK_ENDPOINT"]


def add_margin(pil_img, mode, scaling=1.75):
    # scaling has to be greater than 1.33
    width, height = pil_img.size
    new_width = int(width * scaling)
    new_height = int(height * scaling)
    result = Image.new(pil_img.mode, (new_width, new_height), 0)
    if mode == "center":
        w_margin = int(new_width/2 - width/2)
        h_margin = int(new_height/2 - height/2 )
        result.paste(pil_img, (w_margin, h_margin))
    elif mode == "top left":
        w_margin = int(new_width/3 - width/2)
        h_margin = int(new_height/3 - height/2 )
        result.paste(pil_img, (w_margin, h_margin))
    elif mode == "top right":
        w_margin = int(2*new_width/3 - width/2)
        h_margin = int(new_height/3 - height/2 )
        result.paste(pil_img, (w_margin, h_margin))
    elif mode == "bottom left":
        w_margin = int(new_width/3 - width/2)
        h_margin = int(2*new_height/3 - height/2 )
        result.paste(pil_img, (w_margin, h_margin))
    elif mode == "bottom right":
        w_margin = int(2*new_width/3 - width/2)
        h_margin = int(2*new_height/3 - height/2 )
        result.paste(pil_img, (w_margin, h_margin))
    return result


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


def square_crop(image):
    width, height = image.size
    new_width = width if width<=height else height
    new_height = new_width

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    return image.crop((left, top, right, bottom))


def get_subject(input_img):
    client = Client(OCTOAI_TOKEN)

    inputs = {
        "input": {
            "image": image_to_base64(input_img),
            "mode": "fast"
        }
    }
    response = client.infer(endpoint_url="{}/infer".format(CLIP_ENDPOINT), inputs=inputs)
    caption = response["output"]["description"]
    caption = ", ".join(caption.split(",")[0:1])
    print("CLIP interrogator output: {}".format(caption))

    return caption


def get_prompts(caption, theme, num_prompts=10):
    client = Client(OCTOAI_TOKEN)
    # Ask LLAMA for n subject ideas
    llama_inputs = {
        "model": "llama-2-13b-chat-fp16",
        "messages": [
            {
                "role": "assistant",
                "content": "You are a helpful assistant. Do not acknowledge the request with 'sure'. Be consise."},
            {
                "role": "user",
                "content": "You will create a consise and descriptive bullet list of {} product photography backdrops to feature {} with a {} theme. Use at most 10 words per line max. ".format(num_prompts, caption, theme)
            }
        ],
        "stream": False,
        "max_tokens": 256,
        "presence_penalty": 0,
        "temperature": 0.1,
        "top_p": 0.9
    }
    # Send to LLAMA endpoint and do some post processing on the response stream
    outputs = client.infer(endpoint_url="https://text.octoai.run/v1/chat/completions", inputs=llama_inputs)

    # Get the Llama 2 output
    prompts = outputs.get('choices')[0].get("message").get('content')
    prompt_list = [x.lstrip('0123456789.-•* ') for x in prompts.split('\n')]
    prompt_list.remove("")
    if len(prompt_list) > num_prompts:
        prompt_list = prompt_list[1:1+num_prompts]

    # Print the prompt list
    for prompt in prompt_list:
        print(prompt)

    return prompt_list


def launch_imagen(input_img, caption, prompt_list, position, num_images=4):

    inputs = {
        "input": {
            "original": image_to_base64(input_img)
        }
    }

    # obtain depth mask
    client = Client(OCTOAI_TOKEN)
    response = client.infer(endpoint_url=f"{DEPTH_MASK_ENDPOINT}/infer", inputs=inputs)

    # get the crop mask
    resized_string = response["output"]["resized"]
    crop_mask_string = response["output"]["crop_mask"]
    depth_map_string = response["output"]["masked_depth_map"]
    resized = Image.open(BytesIO(b64decode(resized_string)))
    crop_mask = Image.open(BytesIO(b64decode(crop_mask_string)))
    depth_map = Image.open(BytesIO(b64decode(depth_map_string)))
    # Add margin and rescale
    resized = add_margin(resized, position)
    crop_mask = add_margin(crop_mask, position)
    depth_map = add_margin(depth_map, position)
    resized = rescale_image(resized)
    crop_mask = rescale_image(crop_mask)
    depth_map = rescale_image(depth_map)

    # Get cropped
    im_rgb = resized.convert("RGB")
    cropped = im_rgb.copy()
    cropped.putalpha(crop_mask.convert('L'))

    sdxl_futures = []
    for prompt in prompt_list:
        inputs = {
            "prompt": "professional photo, ({}) surrounded by (({})) . 35mm photograph, film, professional, 4k, highly detailed".format(caption, prompt),
            "negative_prompt": "nsfw, out of focus, bokeh, drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
            "width": 1024,
            "height": 1024,
            "style_preset": "base",
            "num_images": 1,
            "steps": 20,
            "cfg_scale": 7.5,
            "use_refiner": True,
            "controlnet": "depth",
            "controlnet_conditioning_scale": 0.65,
            "controlnet_image": image_to_base64(depth_map),
        }
        # generate image
        for i in range(num_images):
            future = client.infer_async(endpoint_url="https://image.octoai.run/generate/controlnet-sdxl", inputs=inputs)
            sdxl_futures.append({
                "future": future,
                "prompt": prompt
            })

    return sdxl_futures, cropped

def generate_gallery(sdxl_futures, cropped, num_images=4):
    client = Client(OCTOAI_TOKEN)

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    i = 0
    for elem in sdxl_futures:
        future = elem["future"]
        prompt = elem["prompt"]
        while not client.is_future_ready(future):
            time.sleep(0.1)
        result = client.get_future_result(future)
        image_str = result["images"][0]["image_b64"]
        image = Image.open(BytesIO(b64decode(image_str)))
        composite = Image.alpha_composite(
            image.convert('RGBA'),
            crop_centered(cropped, image.size)
        )
        cols[i%len(cols)].image(composite, caption=prompt)
        i += 1

st.set_page_config(layout="wide", page_title="LightBox")

st.write("## LightBox - Powered by OctoAI")

st.write("Welcome to LightBox! Upload a product photo and let the AI do all of the hard work of figuring out ways to showcase your product! At times you may experience some slowness. That's because our OctoAI endpoint is warming up. After one or two uses, the app should be a lot snappier.")

my_upload = st.file_uploader("Upload a product photo", type=["png", "jpg", "jpeg"])

theme = st.text_input("Specify a specific theme", value="Thanksgiving")
position = st.radio("Subject position", options=["center", "top left", "top right", "bottom left", "bottom right"], index=0, horizontal=True)

if my_upload:
    my_upload = Image.open(my_upload)
    # Derive what's in the photo
    caption = get_subject(my_upload)
    st.write("I've identified the subject to be: \n \t{}".format(caption))
    # Some pre-processing
    my_upload = square_crop(my_upload)
    my_upload = rescale_image(my_upload)
    # Derive background ideas from LLAMA2
    prompt_list = get_prompts(caption, theme)
    st.write("Here are 10 suggested backdrops: \n - {}".format( "\n - ".join(prompt_list)))
    # Generate a beautiful gallery
    sdxl_futures, cropped = launch_imagen(my_upload, caption, prompt_list, position)
    generate_gallery(sdxl_futures, cropped)
from octoai.client import Client
import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
import os
import random
import queue


# These need to be set in your environment
OCTOAI_TOKEN = os.environ["OCTOAI_TOKEN"]
BLIP_ENDPOINT = os.environ["BLIP_ENDPOINT"]
LLAMA2_ENDPOINT = os.environ["LLAMA2_ENDPOINT"]
DEPTH_MASK_ENDPOINT = os.environ["DEPTH_MASK_ENDPOINT"]
SDXL_DEPTH_ENDPOINT = os.environ["SDXL_DEPTH_ENDPOINT"]


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
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


def get_subject(my_upload):
    client = Client(OCTOAI_TOKEN)

    input_img = Image.open(my_upload)
    inputs = {
        "input": {
            "image": image_to_base64(input_img)
        }
    }
    response = client.infer(endpoint_url="{}/infer".format(BLIP_ENDPOINT), inputs=inputs)
    caption = response["output"]["caption"]
    print("BLIP output: {}".format(caption))

    return caption


def get_prompts(caption, num_prompts=10):
    client = Client(OCTOAI_TOKEN)
    # Ask LLAMA for n subject ideas
    llama_inputs = {
        "model": "llama-2-7b-chat",
        "messages": [
            {
                "role": "assistant",
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
            {
                "role": "user",
                "content": "Provide a consise bullet list of {} product photographs subjects featuring {}. 10 words per line max.".format(num_prompts, caption)
            }
        ],
        "stream": False,
        "max_tokens": 512
    }
    # Send to LLAMA endpoint and do some post processing on the response stream
    outputs = client.infer(endpoint_url="{}/v1/chat/completions".format(LLAMA2_ENDPOINT), inputs=llama_inputs)

    # Get the Llama 2 output
    prompts = outputs.get('choices')[0].get("message").get('content')
    prompt_list = [x.lstrip('0123456789.-•* ') for x in prompts.split('\n')]
    if len(prompt_list) > num_prompts:
        prompt_list = prompt_list[1:1+num_prompts]

    # Print the prompt list
    for prompt in prompt_list:
        print(prompt)

    return prompt_list


def launch_imagen(my_upload, caption, prompt_list, num_images=4):

    input_img = Image.open(my_upload)
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
            "input": {
                "prompt": "{}".format(prompt),
                "negative_prompt": "nsfw, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
                "image": image_to_base64(depth_map),
                "num_images": 1,
                "num_inference_steps": 20
            }
        }
        # generate image
        for i in range(num_images):
            inputs["input"]["seed"] = random.randint(0, 4096)
            future = client.infer_async(endpoint_url=f"{SDXL_DEPTH_ENDPOINT}/infer", inputs=inputs)
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
    while len(sdxl_futures):
        for elem in sdxl_futures:
            future = elem["future"]
            prompt = elem["prompt"]
            if client.is_future_ready(future):
                result = client.get_future_result(future)
                image_str = result["output"]["images"][0]["base64"]
                image = Image.open(BytesIO(b64decode(image_str)))
                composite = Image.alpha_composite(
                    image.convert('RGBA'),
                    crop_centered(cropped, image.size)
                )
                cols[i%len(cols)].image(composite, caption=prompt)
                i += 1
                sdxl_futures.remove(elem)

st.set_page_config(layout="wide", page_title="LightBox")

st.write("## LightBox - Powered by OctoAI")

my_upload = st.file_uploader("Upload a product photo", type=["png", "jpg", "jpeg"])

if my_upload:
    caption = get_subject(my_upload)
    st.write("I've identified the subject to be: {}".format(caption))
    prompt_list = get_prompts(caption)
    st.write("Here are 10 ideas for product photography: \n - {}".format( "\n - ".join(prompt_list)))
    sdxl_futures, cropped = launch_imagen(my_upload, caption, prompt_list)
    generate_gallery(sdxl_futures, cropped)
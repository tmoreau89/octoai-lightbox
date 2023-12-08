from octoai.client import Client
from streamlit_image_select import image_select
import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
import os
import time

# These need to be set in your environment
OCTOAI_TOKEN = os.environ["OCTOAI_API_TOKEN"]
DEPTH_MASK_ENDPOINT = os.environ["DEPTH_MASK_ENDPOINT"]

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


def get_prompts(client, theme, num_prompts=10):
    # Ask LLAMA for n backdrop ideas
    llama_inputs = {
        "model": "mistral-7b-instruct-fp16",
        "messages": [
            {
                "role": "assistant",
                "content": "You are a helpful assistant. Answer the question straight away with a bullet list of answers. Do not acknowledge the request with 'sure'."},
            {
                "role": "user",
                "content": "You will create a consise and descriptive bullet list of {} fashion photography backdrops on the theme of {}. Use at most 10 words per line max.".format(num_prompts, theme)
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


def obtain_depth_maps_receive(client, depth_map_futures):
    depth_maps = []
    cropped_imgs = []
    for future in depth_map_futures:
        while not client.is_future_ready(future):
            time.sleep(0.1)
        result = client.get_future_result(future)
        # get the crop mask
        resized_string = result["output"]["resized"]
        crop_mask_string = result["output"]["crop_mask"]
        depth_map_string = result["output"]["masked_depth_map"]
        resized = Image.open(BytesIO(b64decode(resized_string)))
        crop_mask = Image.open(BytesIO(b64decode(crop_mask_string)))
        depth_map = Image.open(BytesIO(b64decode(depth_map_string)))
        # Add margin and rescale
        resized = rescale_image(resized)
        crop_mask = rescale_image(crop_mask)
        depth_map = rescale_image(depth_map)
        depth_maps.append(depth_map)
        # Get cropped
        im_rgb = resized.convert("RGB")
        cropped = im_rgb.copy()
        cropped.putalpha(crop_mask.convert('L'))
        cropped_imgs.append(cropped)

    return depth_maps, cropped_imgs


def cycle_backgrounds_launch(client, depth_maps, subject, prompt_list, theme):
    sdxl_futures = []
    for prompt in prompt_list:
        inputs = {
            "prompt": "{} posing in front of {}, {}".format(subject, prompt, theme),
            "negative_prompt": "nsfw, anime, cartoon, drawing, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, unprofessional, blurry",
            "width": 1024,
            "height": 1024,
            "style_preset": "base",
            "num_images": 1,
            "steps": 30,
            "cfg_scale": 7.5,
            "use_refiner": False,
            "high_noise_frac": 0.8,
            "controlnet": "depth_sdxl",
            "controlnet_conditioning_scale": 0.95,
        }
        # generate image
        for img in depth_maps:
            inputs["controlnet_image"] = image_to_base64(img)
            future = client.infer_async(endpoint_url="https://image.octoai.run/generate/controlnet-sdxl", inputs=inputs)

            sdxl_futures.append({
                "future": future,
                "prompt": prompt
            })

    return sdxl_futures


def cycle_backgrounds_receive(client, sdxl_futures, cropped_imgs, num_images=5):
    col1, col2, col3, col4, col5 = st.columns(num_images)
    cols = [col1, col2, col3, col4, col5]

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
            crop_centered(cropped_imgs[i%num_images], image.size)
        )
        cols[i%len(cols)].image(composite, caption=prompt)
        i += 1

def generate_shots_launch(client, subject, product, num_images=5):
    if product == 0:
        # Apple Vision Pro
        lora = {"asset_01he874n9pehcb2m60mn6ftvjr": 0.8}
        prompt_add = "wearing an apple vision pro"
        key_word = "((cpcvision))"
    elif product == 1:
        # Octo Shirt
        lora = {"octoml_shirt_v8": 1}
        prompt_add = "wearing a black tshirt"
        key_word = "((orishirt))"

    SDXL_payload = {
        "prompt": "{}, fashion shot of {} {}. body positive, film, trendy, stylish, professional, highly detailed".format(key_word, subject, prompt_add),
        "negative_prompt": "(visible hands), nsfw, anime, cartoon, drawing, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, unprofessional, blurry",
        "loras": lora,
        "style_preset": "base",
        "width": 1024,
        "height": 1024,
        "num_images": 1,
        "sampler": "DPM_PLUS_PLUS_SDE_KARRAS",
        "use_refiner": True,
        "steps": 30,
        "cfg_scale": 7.5
    }

    # generate image
    sdxl_futures = []
    for i in range(num_images):
        future = client.infer_async(endpoint_url=f"https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
        sdxl_futures.append(future)

    return sdxl_futures


def generate_shots_receive(p_client, d_client, sdxl_futures, num_images=5):
    col1, col2, col3, col4, col5 = st.columns(num_images)
    cols = [col1, col2, col3, col4, col5]

    depth_map_futures = []
    images = []
    i = 0
    for future in sdxl_futures:
        while not p_client.is_future_ready(future):
            time.sleep(0.1)
        result = p_client.get_future_result(future)
        image_str = result["images"][0]["image_b64"]
        # Launch depth map generation on other endpoint
        inputs = {
            "input": {
                "original": image_str
            }
        }
        future = d_client.infer_async(endpoint_url=f"{DEPTH_MASK_ENDPOINT}/infer", inputs=inputs)
        depth_map_futures.append(future)
        # Display images
        image = Image.open(BytesIO(b64decode(image_str)))
        images.append(image)
        cols[i%len(cols)].image(image, caption="Standard background")
        i += 1
    return depth_map_futures, images


st.set_page_config(layout="wide", page_title="OctoStudio")

st.write("## OctoStudio - Powered by OctoAI")

input_image_idx = image_select(
    label="Select a product you wish to showcase",
    images=[
        Image.open("assets/vision_pro.jpeg"),
        Image.open("assets/octo_shirt.jpeg"),
    ],
    captions=["Vision Pro", "OctoML Shirt"],
    use_container_width=False,
    return_value="index"
)

subject = st.text_input("Describe a subject", value="Indian male in his 30s")
theme = st.text_input("Specify a theme for the backdrop", value="Holi")

prod_client = Client(OCTOAI_TOKEN)

gallery_width = 5
gallery_height = 5

if st.button('Generate Gallery!'):


    # Generate original fashion shots (launch)
    sdxl_futures = generate_shots_launch(prod_client, subject, input_image_idx, gallery_width)

    # Obtain prompts in the meanwhile
    prompt_list = get_prompts(prod_client, theme, gallery_height)
    st.write("Here are {} suggested backdrops by Mistral: \n - {}".format(gallery_height, "\n - ".join(prompt_list)))

    # Generate original fashion short (receive)
    dm_futures, images = generate_shots_receive(prod_client, prod_client, sdxl_futures, gallery_width)

    # Get depth maps
    depth_maps, cropped_imgs = obtain_depth_maps_receive(prod_client, dm_futures)

    # Generate gallery of new backgrounds
    start = time.time()
    sdxl_depth_futures = cycle_backgrounds_launch(prod_client, depth_maps, subject, prompt_list, theme)
    cycle_backgrounds_receive(prod_client, sdxl_depth_futures, cropped_imgs, gallery_width)

    end = time.time()
    print("{} images took {}s".format(gallery_width*gallery_height, end-start))

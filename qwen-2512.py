# https://huggingface.co/docs/diffusers/quicktour
import datetime
import torch
import argparse
import logging
import os
from pathlib import Path
from diffusers import DiffusionPipeline

# Setup logging[]
LOG_FILE = "qwen2512.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',  # append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_prompt_from_user():
    parser = argparse.ArgumentParser(description="Generate an image with Qwen-Image-2512")
    parser.add_argument(
        "--prompt", "-p", 
        type=str, 
        help="The prompt to generate the image. If not provided, will be asked interactively."
    )
    args = parser.parse_args()

    if args.prompt:
        return args.prompt
    else:
        print("Enter the image generation prompt (press Enter twice to finish):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line:
                break
            lines.append(line)
        return "\n".join(lines)

# Get dynamic prompt
prompt = get_prompt_from_user()
logging.info(f"Received prompt:\n{prompt}")

# Generate timestamped image filename
date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename = f"qwen2512_image-{date}.png"
logging.info(f"Image will be saved as: {image_filename}")

# Load and configure pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16, device_map="balanced"
)

pipeline.reset_device_map()
pipeline.enable_sequential_cpu_offload()

# Generate image
images = pipeline(
    prompt=prompt,
    num_inference_steps=35,
    guidance_scale=10
).images[0]

# Save and log result
images.save(image_filename)
logging.info(f"Image saved successfully: {image_filename}")
print(f"✅ Image saved: {image_filename}")
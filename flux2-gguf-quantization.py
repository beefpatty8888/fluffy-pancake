
# https://huggingface.co/docs/diffusers/main/quantization/gguf
# https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/41\
# NOTE: the GPU VRAM must be large enough to load this entire quantized model.
# it does not seem to balance between system and GPU memory.

import argparse
import logging
import datetime
from diffusers import Flux2Pipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
import torch

# Setup logging[]
LOG_FILE = "flux2-gguf.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',  # append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_prompt_from_user():
    parser = argparse.ArgumentParser(description="Generate an image with flux2")
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

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="flux2-gguf_image-"+date+".png"
logging.info(f"Image will be saved as: {image_filename}")

transformer = Flux2Transformer2DModel.from_single_file(
    "/home/jack/repos/fluffy-pancake/flux2-dev-Q8_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16
)

pipeline = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
#    device_map="balanced"
).to("cuda")

images = pipeline(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=12,
).images[0]

images.save(image_filename)
logging.info(f"Image saved successfully: {image_filename}")
print(f"✅ Image saved: {image_filename}")
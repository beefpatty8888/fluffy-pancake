
# https://huggingface.co/docs/diffusers/main/quantization/gguf
# https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/41\
# NOTE: the GPU VRAM must be large enough to load this entire quantized model.
# it does not seem to balance between system and GPU memory.

import datetime
from diffusers import Flux2Pipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
import torch


date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="flux2-gguf_image-"+date+".png"

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

prompt = """
A Victorian mansion sits near a lake in Cornwall, England during the golden hour just before sunset.
The lake has a pier with a small rowboat tied to it.
The mansion is surrounded by a forest and hills in the background.
"""

#pipeline.reset_device_map()
#pipeline.enable_model_cpu_offload()
#pipeline.vae.enable_slicing()
#pipeline.enable_sequential_cpu_offload()

images = pipeline(
    prompt=prompt,
    num_inference_steps=35,
    guidance_scale=12,
).images[0]

images.save(image_filename)
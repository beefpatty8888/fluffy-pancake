# https://huggingface.co/docs/diffusers/quicktour
import datetime
import torch
from diffusers import FluxPipeline

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="flux_image-"+date+".png"
#torch.cuda.empty_cache()

pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced")

#pipeline = DiffusionPipeline.from_pretrained(
 ##"black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, device_map="balanced"
#  "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16, device_map="balanced"
#)

prompt = """
cinematic film still of a cat sipping a margarita in the hills of Texas.
cinemascope, moody, epic, gorgeous, film grain
"""
pipeline.reset_device_map()
#pipeline.enable_model_cpu_offload()
pipeline.enable_sequential_cpu_offload()
#pipeline.enable_vae_tiling()
images = pipeline(prompt).images[0]

images.save(image_filename)


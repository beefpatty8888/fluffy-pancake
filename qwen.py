# https://huggingface.co/docs/diffusers/quicktour
import datetime
import torch
from diffusers import DiffusionPipeline

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="qwen_image-"+date+".png"
#torch.cuda.empty_cache()
pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="balanced"
)

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


# https://huggingface.co/blog/flux-2
import datetime
import torch
from diffusers import Flux2Pipeline

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="flux2_image-"+date+".png"
#torch.cuda.empty_cache()

pipeline = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, device_map="balanced")

#pipeline = DiffusionPipeline.from_pretrained(
 ##"black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, device_map="balanced"
#  "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16, device_map="balanced"
#)

#prompt = """
#a cat with a cowboy hat and bandana sipping whiskey in the hills of Texas with a forest in the background. Sunny skies, clouds in the horizon
#epic, gorgeous, film grain
#"""

prompt = """
a woman with a sword and a shield is riding a horse on a trail in the woods. Ahead of her in the distance is a dragon guarding a castle. Rays of sunshine is shining through the clouds. In the background are some hills and a waterfall.
epic, adventurous.
"""
pipeline.reset_device_map()
#pipeline.enable_model_cpu_offload()
pipeline.enable_sequential_cpu_offload()
images = pipeline(
        prompt=prompt,    
        num_inference_steps=50, # 28 is a good trade-off
        guidance_scale=4
).images[0]


images.save(image_filename)


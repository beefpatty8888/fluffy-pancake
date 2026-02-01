# https://huggingface.co/docs/diffusers/quicktour
import datetime
import torch
from diffusers import DiffusionPipeline

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="qwen2512_image-"+date+".png"
#torch.cuda.empty_cache()
pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16, device_map="balanced"
)

#prompt = """
#a rocket with boosters, one on each side, is sitting on a launchpad at night. Searchlights illuminate the rocket. A full moon is up, and the moonlight is reflected on the lake in the background.
#"""

prompt = """
a woman with a sword and a shield is riding a horse on a trail in the woods. Ahead of her in the distance is a dragon guarding a castle. Rays of sunshine is shining through the clouds. In the background are some hills and a waterfall.
"""


pipeline.reset_device_map()
#pipeline.enable_model_cpu_offload()
pipeline.enable_sequential_cpu_offload()
#pipeline.enable_vae_tiling()
images = pipeline(prompt).images[0]

images.save(image_filename)


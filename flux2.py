# https://huggingface.co/blog/flux-2
import datetime
import torch
from diffusers import Flux2Pipeline

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="flux2_image-"+date+".png"
#torch.cuda.empty_cache()

pipeline = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, device_map="balanced")

#prompt = """
#a cat with a cowboy hat and bandana sipping whiskey in the hills of Texas with a forest in the background. Sunny skies, clouds in the horizon
#epic, gorgeous, film grain
#"""

#prompt = """
#a woman with a sword and a shield is riding a horse on a trail in the woods. Ahead of her in the distance is a dragon guarding a castle. Rays of sunshine is shining through the clouds. In the background are some hills and a waterfall.
#epic, adventurous.
#"""

#prompt = """
#A silhouette of a couple dancing the tango under a pavilion at night. The moon is full and the moonlight is reflecting on a lake in the background.
#The pavilion is decorated with lanterns, creating a romantic atmosphere. The park is filled with trees, flowers, benches and a cobblestone pathway.
#"""

#prompt = """
#a cowboy is riding a horse through the prairie at sunrise, herding cattle and longhorn bovine. 
#In the background are some hills and a river reflecting the golden light of the rising sun with a few scattered clouds.
#"""

#prompt = """
#A Victorian mansion sits next to a lake in Cornwall, England during the golden hour just before sunset.
#The lake has a pier with a small rowboat tied to it.
#The mansion is surrounded by a forest and hills in the background.
#"""

prompt = """
A Union Pacific steam locomative train is pulling a line of passanger cars with a red caboose trailing as the final car. The landscape is mountainous and covered in snow during winter.
The train is emitting a plume of white smoke as it travels along the tracks. The surrounding scenery includes snow-covered trees, rocky hills, and a blue sky with cumulo-form clouds.
There are tunnels in the mountains where the train passes through.
A small town and a river can be seen in the distance. 
"""

pipeline.reset_device_map()
#pipeline.enable_model_cpu_offload()
pipeline.vae.enable_slicing()
pipeline.enable_sequential_cpu_offload()
images = pipeline(
        prompt=prompt,    
        num_inference_steps=50, # 28 is a good trade-off
        guidance_scale=4
).images[0]


images.save(image_filename)


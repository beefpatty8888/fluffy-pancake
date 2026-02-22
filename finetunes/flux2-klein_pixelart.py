# https://huggingface.co/artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B

import datetime
import torch
from diffusers import Flux2KleinPipeline

from peft import PeftModel, PeftConfig

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="flux2-klein_image_pixelart-"+date+".png"

pipeline = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16, device_map="cuda")

pipeline.load_lora_weights(
    "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B",
    adapter_name="pixelart"
)

# https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference?weight-scale=finer+control#weight-scale
num_inference_steps = 50
lora_steps = 25
lora_scales = torch.linspace(1.0, 0.5, lora_steps).tolist()
lora_scales += [0.2] * (num_inference_steps - lora_steps + 1)

pipeline.set_adapters("pixelart", lora_scales[0])

def callback(pipeline: Flux2KleinPipeline, step: int, timestep: torch.LongTensor, callback_kwargs: dict):
    pipeline.set_adapters("pixelart", lora_scales[step + 1])
    return callback_kwargs


prompt = """
A samurai is standing in a serene garden with cherry blossoms. 
Behind him is a torri gate with a stone lantern, and a koi pond with colorful fish. Pixel Art, PixArFK
"""

image = pipeline(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=9.0,
    num_inference_steps=num_inference_steps,
#    not yet supported  for the Flux2KleinPipeline
#    cross_attention_kwargs={"scale": 0.5},
    generator=torch.manual_seed(543211),
    callback_on_step_end=callback
).images[0]

image.save(image_filename)
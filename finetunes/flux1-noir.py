# Noir Style Lora finetuning model - https://civitai.com/models/1005742?modelVersionId=1127194
#
#
import datetime
import torch
from diffusers import FluxPipeline, AutoencoderKL

from peft import PeftModel, PeftConfig

date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
image_filename="flux_image_noir-"+date+".png"


# https://github.com/huggingface/diffusers/issues/6815#issuecomment-1925810346
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

pipeline.load_lora_weights(
    '/home/jack/repos/fluffy-pancake/finetunes/Noir-Style-Photography.safetensors',
    adapter_name="noir"
)

# https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference?weight-scale=finer+control#weight-scale
num_inference_steps = 50
lora_steps = 40
lora_scales = torch.linspace(1.0, 0.5, lora_steps).tolist()
lora_scales += [0.2] * (num_inference_steps - lora_steps + 1)

pipeline.set_adapters("noir", lora_scales[0])

def callback(pipeline: FluxPipeline, step: int, timestep: torch.LongTensor, callback_kwargs: dict):
    pipeline.set_adapters("noir", lora_scales[step + 1])
    return callback_kwargs

prompt="""
Photograph of a jazz band playing in a dimly lit club. The band consists of a saxophonist, a pianist, and a drummer.
A woman in a provocative dress is sitting at a table in the foreground. A cocktail glass with a cherry is on the table.
noirstyle. monochrome, black and white, ultra detailed textures and skin
"""


negative_prompt = (
"text, watermark, signature, cartoon, anime, illustration, painting, drawing, low quality, blurry"
)


images = pipeline(
    prompt = prompt, 
    height=1024,
    width=1024,
    #negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    guidance_scale=9,
    # https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference?weight-scale=simple+use+case#weight-scale
    # https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/flux#diffusers.FluxPipeline.__call__.joint_attention_kwargs
    #joint_attention_kwargs={"scale": 0.5}
    generator=torch.Generator("cuda").manual_seed(123456712),
    callback_on_step_end=callback
).images[0]

images.save(image_filename)


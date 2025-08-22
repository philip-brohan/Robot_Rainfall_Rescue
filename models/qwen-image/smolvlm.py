#!/usr/bin/env python

# Generate an image using the Qwen-Image model
# Make an anthropomorphised picture of the SmolVLM model

from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)


# Generate image
prompt = """create an anthropomorphised robot version of the SmolVLM model from Hugging Face as a science tool, Use the following concepts in the drawing: Small & Efficient Look: SmolVLM is a lightweight model, so the robot should be compact and nimble — maybe around child-sized with a sleek, minimalist design. Hugging Face Theme: It should have a friendly expression, with the signature 🤗 Hugging Face emoji-style face as inspiration. One hand shaped like a camera (vision capability). The other like a pen or pencil (for language interaction). The letters "SmolVLM" subtly integrated on its chest or shoulder."""

negative_prompt = (
    " "  # using an empty string if you do not have specific concept to remove
)


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

image.save("outputs/smolvlm.png")

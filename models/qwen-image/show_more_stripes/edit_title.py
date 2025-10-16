#!/usr/bin/env python

# Generate an image using the Qwen-Image model
# Make astick-figue illustration for the show-more-stripes ppt

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

# Style prompt - added to all prompts
style_prompt = [
    "A cartoon. Simple and clean style, , white background, minimalistic design, high contrast, vector art.",
    "Text in the image drawn as black on white. Black and white only.",
]

style_negative_prompt = [
    "Details, colors, shading, textures, backgrounds, white on black text"
]

# Define some characters to use in the images
person1 = (
    "A man, short and a little tubby, about 50 years old. "
    "With short, thinning, grey hair, glasses, wearing a striped shirt and dark trousers."
)
person2 = (
    "A tall, thin, man, about 30 years old, with short dark hair, wearing overalls."
)
person3 = "A young man with a bushy beard, wearing a sweater and jeans."

# Specify the set of images. For each, we need a name, aspect ratio, prompt and negative prompt.
images = {
    "paint-roller": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, facing away from us, looking up and to the left." % person2,
                "Holding a paint roller on a long pole, stretching up high and a bit to his left.",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "thinking-of-title": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, facing left, looking up and to the left, thinking, with his chin in his hand."
                % person1,
                "Saying 'Stripes, Speckles, Splotches, ...'.",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "detailed-explanation": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, facing left, leaning on a vertical wall behind him."
                % person3,
                "Saying 'What we really need is a detailed explanation'.",
                "Show the whole figure, head to toes.",
            ]
        ),
        "negative_prompt": " ".join(["Shadow"]),
    },
    "classic": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, facing to the left, looking down, hands raised"
                % person1,
                "saying enthusiastically 'A classic'.",
                "Show the whole figure, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "boring": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, looking down and left, looking grumpy." % person3,
                "Saying 'It's a bit boring.'",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "latitude-resolved": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, looking right, looking enthusiastic." % person1,
                "Saying 'Resolve the latitude: North pole at the top, South pole at the bottom.'",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "monthly-resolved": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, looking right, looking enthusiastic." % person1,
                "Saying 'One stripe per month, rather than per year.'",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "longitude-sampled": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, looking left, looking doubtful." % person1,
                "Saying 'Sample in longitude, rather than averaging.'",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "seated-etcw": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, siting on a plain rectangular block, looking right, looking rebellious."
                % person3,
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "chide-etcw": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, looking left, looking angry." % person1,
                "Saying 'Get your feet out of the early 20th century warming.'",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "chide-missing": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, looking left, looking mischevious." % person3,
                "Saying 'I suppose the black is where you havent got the observations yet.'",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "fill-missing": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, facing away from us, looking up and to the left."
                % person1,
                "Carying an open book in one hand, and a painbrush in the other.",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "chide-incomplete": {
        "aspect": aspect_ratios["9:16"],
        "prompt": " ".join(
            [
                "%s, standing, looking left, looking mischevious." % person3,
                "Saying 'I suppose the black is where you havent got the observations yet.'",
                "Show the whole man, head to toes.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "fill-missing": {
        "aspect": aspect_ratios["1:1"],
        "prompt": " ".join(
            [
                "%s, sitting at a desk, with a laptop open in front of them." % person2,
                "Seen from above and to their left",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "extinguisher": {
        "aspect": aspect_ratios["1:1"],
        "prompt": " ".join(
            [
                "%s, discharging a fire extinguisher pointing down and to the left"
                % person2,
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
    "caveman": {
        "aspect": aspect_ratios["1:1"],
        "prompt": " ".join(
            [
                "%s, sitting on the ground, warming his hands over a small fire."
                % person2,
                "Wearing a bowler hat and a fur tunic.",
            ]
        ),
        "negative_prompt": " ".join([" "]),
    },
}

for key, value in images.items():
    width, height = value["aspect"]
    prompt = [value["prompt"]]
    negative_prompt = [value["negative_prompt"]]

    image = pipe(
        prompt=" ".join(style_prompt + prompt),
        negative_prompt=" ".join(style_negative_prompt + negative_prompt),
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]

    image.save(f"outputs/{key}.png")

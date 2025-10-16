#!/usr/bin/env python

# Test the effect of the Cosmos tokenizer on Ed's monthly rainfall images

import os
import sys
import numpy as np
import random
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import torch
import matplotlib.pyplot as plt
import argparse

from rainfall_rescue.utils.pairs import get_index_list, load_pair


sys.path.append("%s/models/Cosmos/Cosmos-Tokenizer" % os.getenv("PYTHONPATH"))
from cosmos_tokenizer.image_lib import ImageTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="Cosmos-Tokenizer-CI16x16",
)
parser.add_argument(
    "--label",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
args = parser.parse_args()
if args.label is None:
    args.label = random.choice(get_index_list())

# load the image/data pair
original_image, csv = load_pair(args.label)


encoder = ImageTokenizer(
    checkpoint_enc=f"{os.getenv('HF_HOME')}/pretrained_ckpts/{args.model_id}/encoder.jit"
)
decoder = ImageTokenizer(
    checkpoint_dec=f"{os.getenv('HF_HOME')}/pretrained_ckpts/{args.model_id}/decoder.jit"
)


def image_to_tensor(image, contrast=None):
    """Convert an image to a tensor."""
    it = image.copy()
    if contrast is not None:
        enhancer = ImageEnhance.Contrast(it)
        it = enhancer.enhance(contrast)
    tensor = (
        torch.from_numpy(np.array(it).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        / 255.0
    )
    return tensor


def tensor_to_image(tensor):
    """Convert a tensor back to an image."""
    tensor = torch.clamp(tensor, 0.0, 1.0)
    image_array = (
        (tensor * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    )
    return Image.fromarray(image_array)


def autoencode(image_tensor, encoder, decoder):
    """Autoencode an image using the provided encoder and decoder."""
    input_tensor = image_tensor.to("cuda").to(torch.bfloat16)

    # Encode the image
    encoding = encoder.encode(input_tensor)

    # Decode the latent representation
    reconstructed_tensor = decoder.decode(encoding[0])

    return tensor_to_image(reconstructed_tensor)


# Plot the original and autoencoded images side-by-side
def plot_images(original_image, reconstructed_image, file_name="outputs/test.png"):
    """Plot original and reconstructed images side by side."""
    aspect = original_image.width / original_image.height
    plt.figure(figsize=(20 * aspect * 2, 20))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image)
    plt.axis("off")

    plt.savefig(file_name)


# Effect of varying image size
def test_image_size_effects():
    """Test the effect of varying image sizes on autoencoding."""
    contrasts = [None, 0.5, 1.5, 2.0, 2.5]

    for contrast in contrasts:
        resized_image = original_image.copy()
        if contrast is not None:
            enhancer = ImageEnhance.Contrast(resized_image)
            resized_image = enhancer.enhance(contrast)
        image_tensor = image_to_tensor(original_image, contrast=contrast)
        reconstructed_image = autoencode(image_tensor, encoder, decoder)
        if contrast is None:
            contrast = "original"
        else:
            contrast = "%2d" % int(contrast * 10)
        plot_images(
            resized_image, reconstructed_image, file_name=f"outputs/test_{contrast}.png"
        )


if __name__ == "__main__":
    # Test the effect of varying image sizes
    test_image_size_effects()

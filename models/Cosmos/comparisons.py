#!/usr/bin/env python

import os
import sys
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
import matplotlib.pyplot as plt

sys.path.append("%s/models/Cosmos/Cosmos-Tokenizer" % os.getenv("PYTHONPATH"))
from cosmos_tokenizer.image_lib import ImageTokenizer

original_image_url = (
    "https://brohan.org/AI_daily_precip/_images/TYRain_1941-1950_25_pt1-10.jpg"
)

model_name = "Cosmos-Tokenizer-CI16x16"
encoder = ImageTokenizer(
    checkpoint_enc=f"{os.getenv('HF_HOME')}/pretrained_ckpts/{model_name}/encoder.jit"
)
decoder = ImageTokenizer(
    checkpoint_dec=f"{os.getenv('HF_HOME')}/pretrained_ckpts/{model_name}/decoder.jit"
)


def load_image_from_url(url):
    """Load an image from a URL."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    return Image.open(BytesIO(response.content))


def image_to_tensor(image, pixels=512):
    """Convert an image to a tensor."""
    it = image.copy()
    it.thumbnail((pixels, pixels), Image.Resampling.LANCZOS)
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
    plt.figure(figsize=(10 * aspect * 2, 10))

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
    sizes = [64, 128, 256, 512, 1024, 2048]
    original_image = load_image_from_url(original_image_url)

    for size in sizes:
        resized_image = original_image.copy()
        resized_image.thumbnail((size, size), Image.Resampling.LANCZOS)
        image_tensor = image_to_tensor(original_image, pixels=size)
        reconstructed_image = autoencode(image_tensor, encoder, decoder)
        plot_images(
            resized_image, reconstructed_image, file_name=f"outputs/test_{size}.png"
        )


if __name__ == "__main__":
    # Test the effect of varying image sizes
    test_image_size_effects()

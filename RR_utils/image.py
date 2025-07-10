# Functions to cut up images (for tokenization) and to stitch them together again

from PIL import Image
import numpy as np


# Cut an image into square blocks of a specified size
#  With padding where necessary.
# def cut_image(image, block_size):
#     width, height = image.size
#     blocks = []

#     # Calculate the number of blocks in each dimension
#     num_blocks_x = (width + block_size - 1) // block_size
#     num_blocks_y = (height + block_size - 1) // block_size

#     # Create blocks with padding if necessary
#     for i in range(num_blocks_y):
#         for j in range(num_blocks_x):
#             left = j * block_size
#             upper = i * block_size
#             right = min(left + block_size, width)
#             lower = min(upper + block_size, height)

#             # Create a new image for the block
#             cropped = image.crop((left, upper, right, lower))
#             # If the block is smaller than the specified size, pad it
#             if cropped.width < block_size or cropped.height < block_size:
#                 block = Image.new(
#                     image.mode, (block_size, block_size), color="white"
#                 )  # White padding
#                 block.paste(cropped, (0, 0))
#             else:
#                 block = cropped

#             blocks.append(block)

#     return blocks


def cut_image(image, block_size, overlap=0.0):
    """
    Cut an image into square blocks of a specified size with adjustable overlap.

    Args:
        image (PIL.Image): The input image.
        block_size (int): The size of each block (in pixels).
        overlap (float): Fraction of block size to overlap between blocks (0.0 to 1.0).

    Returns:
        list: List of image blocks.
    """
    width, height = image.size
    blocks = []

    # Calculate the step size based on overlap
    step_size = int(block_size * (1 - overlap))

    # Create blocks with padding if necessary
    for upper in range(0, height, step_size):
        for left in range(0, width, step_size):
            right = min(left + block_size, width)
            lower = min(upper + block_size, height)

            # Create a new image for the block
            cropped = image.crop((left, upper, right, lower))
            # If the block is smaller than the specified size, pad it
            if cropped.width < block_size or cropped.height < block_size:
                block = Image.new(
                    image.mode, (block_size, block_size), color="white"
                )  # White padding
                block.paste(cropped, (0, 0))
            else:
                block = cropped

            blocks.append(block)

    return blocks

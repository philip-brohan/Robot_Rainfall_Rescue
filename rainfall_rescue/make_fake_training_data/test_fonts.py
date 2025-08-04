#!/usr/bin/env python

# Print a test for each font - just to make sure they are suitable

from fonts import fontNames

test_string = ": The quick brown fox jumps over the lazy dog 1234567890."

# Matplotlib figure - were going to use ax.text to plot a string
#  with each of the available fonts
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(fontNames) + 1)
ax.axis("off")

for i, font_name in enumerate(fontNames):
    try:
        ax.text(
            0.5,
            len(fontNames) - i,
            font_name + test_string,
            fontdict={
                "family": font_name,
            },
            ha="center",
            va="center",
            fontsize=12,
        )
    except Exception as e:
        print(f"Error loading font {font_name}: {e}")

fig.savefig("fonts.png")

#!/usr/bin/env python

# Get font names for all the available fonts
import matplotlib.font_manager
from fontTools.ttLib import TTFont
import string

ascii_chars = string.ascii_letters + string.digits + string.punctuation

# Get a list of font file paths
font_files = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")


def font_supports_ascii(font_path):
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        for char in ascii_chars:
            if ord(char) not in cmap:
                return False
        return True
    except Exception:
        return False


# Filter fonts that support all ASCII characters
ascii_fonts = [f for f in font_files if font_supports_ascii(f)]
# print(ascii_fonts)

# Get a list of font names
font_names = []
for file in ascii_fonts:
    try:
        font_properties = matplotlib.font_manager.FontProperties(fname=file)
        font_names.append(font_properties.get_name())
    except Exception as e:
        print(f"Error processing font file {file}: {e}")
        continue

font_names = sorted(set(font_names))  # Remove duplicates

print(font_names)

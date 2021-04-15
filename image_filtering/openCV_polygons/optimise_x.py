# Functions for layout analysis

import os
import sys

import cv2
import numpy as np
import random

import scipy.optimize

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# Convert image to BW
def imageToBW(image):
    # To greyscale
    pImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # Gaussian blur to denoise
    pImage = cv2.GaussianBlur(pImage, (9, 9), 0)
    # Adaptive threshold to convert to BW
    pImage = cv2.adaptiveThreshold(
        pImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2
    )
    pImage = cv2.bitwise_not(pImage)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
    pImage = cv2.dilate(pImage, kernel)
    return pImage


# Find line segments in the processed image
def imageToLines(image):
    fld = cv2.ximgproc.createFastLineDetector(25, 0.141, 50, 50, 0, False)
    lines = fld.detect(image.copy())

    # Find the nearly-vertical lines in the centre of the image
    nv = []
    for line in lines:
        if max(line[0][1], line[0][3]) > 1200:
            continue
        if min(line[0][1], line[0][3]) < 200:
            continue
        if (abs(line[0][0] - line[0][2]) / abs(line[0][1] - line[0][3])) < 0.1:
            nv.append(line)
    vlines = np.array(nv)

    # Find the nearly-horizontal lines in the centre of the image
    nv = []
    for line in lines:
        if max(line[0][1], line[0][3]) > 1200:
            continue
        if min(line[0][1], line[0][3]) < 300:
            continue
        if (abs(line[0][0] - line[0][2]) / abs(line[0][1] - line[0][3])) > 10:
            nv.append(line)
    hlines = np.array(nv)

    # Find grid-spaced points in the x direction

    return {"vertical": vlines, "horizontal": hlines}


# Optimisation target function - how well does a given spacing and offset fit the data
def fitVlines(x):
    offset = x[0]
    space = x[1]
    # Sum over each of the 11 boundary lines
    count = 0
    result = 0
    for lidx in range(11):
        # find all the x points in the region of the line
        lx = offset + space * lidx
        lxmin = lx - space * 0.6
        lxmax = lx + space * 0.6
        xsl = np.where(np.logical_and(xs > lxmin, xs <= lxmax))
        if (len(xsl[0])) > 0:
            try:
                xss = xs[xsl]
            except Exception:
                print(xs)
                print(xsl)
                sys.exit(1)
            xsl = np.sum((xss - lx) ** 2)
            result += np.sqrt(np.sum(xsl)) / len(xss)
            count += 1
        else:
            result += 1
            count += 1
    return result / count


# Optimisation function - find the spacing and offset with best fit
def findOS(vlines):
    # Put the x points into a single array
    global xs  # Needed by fitVlines and I can't work out how to pass it as argument
    xs = []
    for line in vlines:
        xs.append(line[0][0])
        xs.append(line[0][2])
    xs = np.array(xs)
    # Set initial bounds for spacing and offset
    offset_bounds = (190, 240)
    space_bounds = (54, 74)
    # Small search domain, so brute-force offset and space
    result = scipy.optimize.brute(fitVlines, (offset_bounds, space_bounds))
    return result


# Overplot the line endpoints
def overplotPoints(ax, pLines):
    for lsi in range(len(pLines["vertical"])):
        ls = pLines["vertical"][lsi]
        ax.add_patch(
            Circle(
                (ls[0][0], ls[0][1]),
                radius=5,
                facecolor=(1, 0, 0, 1),
                edgecolor=(1, 0, 0, 1),
                alpha=1,
                zorder=200,
            )
        )
        ax.add_patch(
            Circle(
                (ls[0][2], ls[0][3]),
                radius=5,
                facecolor=(1, 0, 0, 1),
                edgecolor=(1, 0, 0, 1),
                alpha=1,
                zorder=200,
            )
        )


# Overplot the line segment fit
def overplotFit(ax, fit):
    for lsi in range(11):
        ax.add_line(
            Line2D(
                [fit[0] + fit[1] * lsi, fit[0] + fit[1] * lsi],
                [200, 1200],
                linewidth=2,
                color="blue",
                zorder=150,
            )
        )


# Overplot the line segments
def overplotLines(ax, pLines):
    lcm = matplotlib.cm.get_cmap("hsv", len(pLines["vertical"]))
    for lsi in range(len(pLines["vertical"])):
        ls = pLines["vertical"][lsi]
        col = None
        ax.add_line(
            Line2D(
                [ls[0][0], ls[0][2]],
                [ls[0][1], ls[0][3]],
                linewidth=3,
                color="red",  # lcm(lsi),
                zorder=200,
            )
        )
    lcm = matplotlib.cm.get_cmap("hsv", len(pLines["horizontal"]))
    for lsi in range(len(pLines["horizontal"])):
        ls = pLines["horizontal"][lsi]
        col = None
        ax.add_line(
            Line2D(
                [ls[0][0], ls[0][2]],
                [ls[0][1], ls[0][3]],
                linewidth=3,
                color="red",  # lcm(lsi),
                zorder=200,
            )
        )

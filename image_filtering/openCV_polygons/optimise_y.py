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
    # Filter out lines at heights where the lines don't cover most
    # of the image width
    nv = []
    for h in range(300, 1200, 5):
        lc = np.where(np.logical_and(hlines[:, 0, 1] > h - 5, hlines[:, 0, 1] < h + 5))
        if len(lc[0]) == 0:
            continue
        lcl = hlines[lc]
        if np.min(lcl[:, 0, 0]) < 200 and np.max((lcl[:, 0, 0]) > 800):
            for line in lcl:
                nv.append(line)
    hlines = np.array(nv)

    # Find grid-spaced points in the x direction

    return {"vertical": vlines, "horizontal": hlines}


# Optimisation target function - how well do a given offset and spacings fit the data
def fitHlines(x):
    offset = x[0]
    spaceT = x[1]  # Height of totals row
    spaceM = x[1]  # Height of monthly data
    spaceY = x[1]  # Height of years row
    # Sum over each of the 4 boundary lines
    count = 0
    result = 0
    lRange = 5
    # Want lots of points on the lines
    for ly in (
        offset,
        offset - spaceT,
        offset - spaceT - spaceM,
        offset - spaceT - spaceM - spaceY,
    ):
        # find all the x points in the region of the line
        lymin = ly - lRange
        lymax = ly + lRange
        ysl = np.where(np.logical_and(ys > lymin, ys <= lymax))
        result += len(ysl[0])
    # also want few points between the lines
    lymax = offset + lRange
    lymin = offset - spaceT - spaceM - spaceY - lRange
    ysl = np.where(np.logical_and(ys > lymin, ys <= lymax))
    result -= len(ysl[0]) - result
    return result * -1


#        if (len(ysl[0])) > 0:
#            try:
#                yss = ys[ysl]
#            except Exception:
#                print(ys)
#                print(ysl)
#                sys.exit(1)
#            ysl = np.sum((yss - ly) ** 2)
#            result += np.sqrt(np.sum(ysl)) / len(yss)
#            count += 1
#        else:
#            result += 100
#            count += 1
#    return result / count


# Optimisation function - find the spacing and offset with best fit
def findOS(hlines):
    # Put the y points into a single array
    global ys  # Needed by fitHlines and I can't work out how to pass it as argument
    ys = []
    for line in hlines:
        ys.append(line[0][1])
        ys.append(line[0][3])
    ys = np.array(ys)
    # Set initial bounds for spacing and offset
    offset_bounds = (800, 1200)
    T_bounds = (44, 84)
    M_bounds = (450, 690)
    Y_bounds = (34, 74)
    # Small search domain, so brute-force offset and spaces
    result = scipy.optimize.brute(
        fitHlines, (offset_bounds, T_bounds, M_bounds, Y_bounds)
    )
    return result


# Overplot the line endpoints
def overplotPoints(ax, pLines):
    for lsi in range(len(pLines["horizontal"])):
        ls = pLines["horizontal"][lsi]
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
    ax.add_line(
        Line2D(
            [50, 974],
            [fit[0], fit[0]],
            linewidth=2,
            color="blue",
            zorder=150,
        )
    )
    ax.add_line(
        Line2D(
            [50, 974],
            [fit[0] - fit[1], fit[0] - fit[1]],
            linewidth=2,
            color="blue",
            zorder=150,
        )
    )
    ax.add_line(
        Line2D(
            [50, 974],
            [fit[0] - fit[1] - fit[2], fit[0] - fit[1] - fit[2]],
            linewidth=2,
            color="blue",
            zorder=150,
        )
    )
    ax.add_line(
        Line2D(
            [50, 974],
            [fit[0] - fit[1] - fit[2] - fit[3], fit[0] - fit[1] - fit[2] - fit[3]],
            linewidth=2,
            color="blue",
            zorder=150,
        )
    )

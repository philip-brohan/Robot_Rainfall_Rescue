# Functions for layout analysis

import os
import sys

import cv2
import numpy as np
import random

import jenkspy

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

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
    fld = cv2.ximgproc.createFastLineDetector(25, 0.141, 50, 50, 0, True)
    lines = fld.detect(image.copy())

    # Find the nearly-vertical lines in the centre of the image
    nv = []
    for line in lines:
        if max(line[0][1], line[0][3]) > 1200:
            continue
        if min(line[0][1], line[0][3]) < 300:
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

    # Cluster the lines in the x direction
    xs = []
    for line in vlines:
        xs.append(line[0][0])
        xs.append(line[0][2])

    jbx = jenkspy.jenks_breaks(xs, nb_class=13)

    # Replace each vertical cluster with a single line
    # x is cluster mean of x
    # y is cluster max & min of y
    nvlines = []
    for d in range(len(jbx)):
        count = 0
        mx = 0
        nvlines.append([[None, None, None, None]])
        for ls in vlines:
            if ls[0][0] > jbx[d] and ls[0][0] < jbx[d + 1]:
                count += 1
                mx += ls[0][0] + ls[0][2]
                if nvlines[d][0][1] is None or ls[0][1] < nvlines[d][0][1]:
                    nvlines[d][0][1] = ls[0][1]
                if nvlines[d][0][3] is None or ls[0][3] > nvlines[d][0][3]:
                    nvlines[d][0][3] = ls[0][3]
            if count > 0:
                nvlines[d][0][0] = mx / (count * 2)
            else:
                nvlines[d][0][0] = 0
            nvlines[d][0][2] = nvlines[d][0][0]
    vlines = np.array(nvlines)
    # Prune missing clusters and short lines and adjust full lines to more than full length
    nvmin = None
    nvmax = None
    for ls in vlines:
        if ls[0][0] == 0:
            continue  # No lines in cluster
        if nvmin is None or nvmin > ls[0][1]:
            nvmin = ls[0][1]
        if nvmin > ls[0][3]:
            nvmin = ls[0][3]
        if nvmax is None or nvmax < ls[0][1]:
            nvmax = ls[0][1]
        if nvmax < ls[0][3]:
            nvmax = ls[0][3]
    nvlines = []
    for ls in vlines:
        if ls[0][0] == 0:
            continue
        if (ls[0][3] - ls[0][1]) < 300:
            continue
        ls[0][1] = nvmin - 20
        ls[0][3] = nvmax + 20
        nvlines.append(ls)
    vlines = np.array(nvlines)

    # Cluster the lines in the y direction
    ys = []
    for line in hlines:
        ys.append(line[0][1])
        ys.append(line[0][3])

    jby = jenkspy.jenks_breaks(ys, nb_class=8)

    # Replace each horizontal cluster with a single line
    # y is cluster mean of y
    # x is cluster max & min of x
    nhlines = []
    for d in range(len(jby)):
        count = 0
        my = 0
        nhlines.append([[None, None, None, None]])
        for ls in hlines:
            if ls[0][1] > jby[d] and ls[0][1] < jby[d + 1]:
                count += 1
                my += ls[0][1] + ls[0][3]
                if nhlines[d][0][0] is None or ls[0][0] < nhlines[d][0][0]:
                    nhlines[d][0][0] = ls[0][0]
                if nhlines[d][0][2] is None or ls[0][2] > nhlines[d][0][2]:
                    nhlines[d][0][2] = ls[0][2]
            if count > 0:
                nhlines[d][0][1] = my / (count * 2)
            else:
                nhlines[d][0][1] = 0
            nhlines[d][0][3] = nhlines[d][0][1]
    hlines = np.array(nhlines)
    # Prune missing clusters and short lines and adjust full lines
    #  to more than full length
    nhmin = None
    nhmax = None
    for ls in hlines:
        if ls[0][1] == 0:
            continue  # No lines in cluster
        if nhmin is None or nhmin > ls[0][0]:
            nhmin = ls[0][0]
        if nhmin > ls[0][2]:
            nhmin = ls[0][2]
        if nhmax is None or nhmax < ls[0][0]:
            nhmax = ls[0][0]
        if nhmax < ls[0][2]:
            nhmax = ls[0][2]
    nhlines = []
    for ls in hlines:
        if ls[0][1] == 0:
            continue
        if (ls[0][2] - ls[0][0]) < 300:
            continue
        ls[0][0] = nhmin - 20
        ls[0][2] = nhmax + 20
        nhlines.append(ls)
    hlines = np.array(nhlines)
    return {"vertical": vlines, "horizontal": hlines}


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

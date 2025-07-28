# Functions for placing and rotating parts of the page

import math


# Rotate by angle degrees clockwise
def gRotate(self, point, angle=None, origin=None):
    if angle is None:
        angle = self.rotate
    if angle == 0:
        return point
    if origin is None:
        origin = gCentre(self)
    ox, oy = origin[0] * self.pageWidth, origin[1] * self.pageHeight
    px, py = point[0] * self.pageWidth, point[1] * self.pageHeight
    angle = math.radians(angle) * -1
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx / self.pageWidth, qy / self.pageHeight


# Centre point of shifted grid
def gCentre(self):
    return (
        0.5 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.43,
        0.525 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.2,
    )


# Corners of grid
def topLeft(self):
    return (
        0.1 + self.xshift / self.pageWidth,
        0.705 + self.yshift / self.pageHeight,
    )


def topRight(self):
    return (
        0.96 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.86,
        0.705 + self.yshift / self.pageHeight,
    )


def bottomLeft(self):
    return (
        0.1 + self.xshift / self.pageWidth,
        0.225 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.4,
    )


def bottomRight(self):
    return (
        0.96 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.86,
        0.225 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.4,
    )


# Point fraction x along top of grid
def topAt(self, x):
    return (
        topRight(self)[0] * x + topLeft(self)[0] * (1 - x),
        topRight(self)[1] * x + topLeft(self)[1] * (1 - x),
    )


# Fraction x along bottom of grid
def bottomAt(self, x):
    return (
        bottomRight(self)[0] * x + bottomLeft(self)[0] * (1 - x),
        bottomRight(self)[1] * x + bottomLeft(self)[1] * (1 - x),
    )


# Fraction y of way up left side
def leftAt(self, y):
    return (
        topLeft(self)[0] * y + bottomLeft(self)[0] * (1 - y),
        topLeft(self)[1] * y + bottomLeft(self)[1] * (1 - y),
    )


# Fraction y of way up right side
def rightAt(self, y):
    return (
        topRight(self)[0] * y + bottomRight(self)[0] * (1 - y),
        topRight(self)[1] * y + bottomRight(self)[1] * (1 - y),
    )


# Apply x and y offsets in rotated coordinates
def gOffset(self, point, xoffset=0, yoffset=0):
    return gRotate(self, [point[0] + xoffset.point[1] + yoffset], self.rotate, point)

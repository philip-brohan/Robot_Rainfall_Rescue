# Class encapsulating a single data cell from a fake ten-year rainfall data image

import os
import math
import random
import pickle
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


class tyrCell:
    def __init__(self, opdir, docn, **kwargs):
        self.opdir = opdir
        self.docn = docn

        # Parameters defining the image geometry
        self.pageWidth = 64
        self.pageHeight = 64
        self.xscale = 1.0
        self.yscale = 1.0
        self.xshift = 0.0  # pixels, +ve right
        self.yshift = 0.0  # pixels, +ve up
        self.rotate = 0.0  # degrees clockwise
        self.linewidth = 1.0
        self.bgcolour = (0.9, 0.9, 0.9)
        self.fgcolour = (0.0, 0.0, 0.0)
        self.boxHeight = 0.66  # Fractional height of data box
        self.boxWidth = 0.66  # Fractional width of data box
        self.fontSize = 12
        self.fontFamily = "Arial"
        self.fontStyle = "normal"
        self.fontWeight = "normal"
        # Noise parameters
        self.jitterFontSize = 0.0
        self.jitterFontRotate = 0.0
        self.jitterGridPoints = 0.0
        self.jitterLineWidth = 0.0

        self.generateNumbers()

        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    setattr(self, key, value)
            else:
                raise ValueError("No parameter %s" % key)

    def dumpState(self):
        if not os.path.isdir("%s/meta" % self.opdir):
            os.makedirs("%s/meta" % self.opdir)
        with open("%s/meta/%s.pkl" % (self.opdir, self.docn), "wb") as pf:
            pickle.dump(vars(self), pf)

    def makeImage(self):
        fig = Figure(
            figsize=(self.pageWidth / 100, self.pageHeight / 100),
            dpi=100,
            facecolor="white",
            edgecolor="black",
            linewidth=0.0,
            frameon=False,
            subplotpars=None,
            tight_layout=None,
        )
        canvas = FigureCanvas(fig)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.set_xlim([0, 1])
        ax_full.set_ylim([0, 1])
        ax_full.set_axis_off()

        # Paint the background
        ax_full.add_patch(
            matplotlib.patches.Rectangle(
                (0, 0), 1, 1, fill=True, facecolor=self.bgcolour
            )
        )

        # Draw the figure
        self.drawBox(ax_full)
        self.drawNumbers(ax_full)
        if not os.path.isdir("%s/images" % self.opdir):
            os.makedirs("%s/images" % self.opdir)
        fig.savefig("%s/images/%s.png" % (self.opdir, self.docn))

    # Must call makeImage before calling this
    #  (bad design but not worth fixing).
    def makeNumbers(self):
        if not os.path.isdir("%s/numbers" % self.opdir):
            os.makedirs("%s/numbers" % self.opdir)
        with open("%s/numbers/%s.pkl" % (self.opdir, self.docn), "wb") as pf:
            pickle.dump(self.rdata, pf)

    # Everything below this is an internal function you should not have to call directly

    # Calculate the grid geometry

    # Rotate by angle degrees clockwise
    def gRotate(self, point, angle=None, origin=None):
        if angle is None:
            angle = self.rotate
        if angle == 0:
            return point
        if origin is None:
            origin = self.gCentre()
        ox, oy = origin[0] * self.pageWidth, origin[1] * self.pageHeight
        px, py = point[0] * self.pageWidth, point[1] * self.pageHeight
        angle = math.radians(angle) * -1
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx / self.pageWidth, qy / self.pageHeight

    # Centre point of shifted box
    def gCentre(self):
        return (
            0.5 + self.xshift / self.pageWidth,
            0.5 + self.yshift / self.pageHeight,
        )

    # Corners of box - before any rotation
    def topLeft(self):
        return (
            (1 - self.boxWidth) / 2 + self.xshift / self.pageWidth,
            1 - (1 - self.boxHeight) / 2 + self.yshift / self.pageHeight,
        )

    def topRight(self):
        return (
            1 - (1 - self.boxWidth) / 2 + self.xshift / self.pageWidth,
            1 - (1 - self.boxHeight) / 2 + self.yshift / self.pageHeight,
        )

    def bottomLeft(self):
        return (
            (1 - self.boxWidth) / 2 + self.xshift / self.pageWidth,
            (1 - self.boxHeight) / 2 + self.yshift / self.pageHeight,
        )

    def bottomRight(self):
        return (
            1 - (1 - self.boxWidth) / 2 + self.xshift / self.pageWidth,
            (1 - self.boxHeight) / 2 + self.yshift / self.pageHeight,
        )

    # Point fraction x along top of grid
    def topAt(self, x):
        return (
            self.topRight()[0] * x + self.topLeft()[0] * (1 - x),
            self.topRight()[1] * x + self.topLeft()[1] * (1 - x),
        )

    # Fraction x along bottom of grid
    def bottomAt(self, x):
        return (
            self.bottomRight()[0] * x + self.bottomLeft()[0] * (1 - x),
            self.bottomRight()[1] * x + self.bottomLeft()[1] * (1 - x),
        )

    # Fraction y of way up left side
    def leftAt(self, y):
        return (
            self.topLeft()[0] * y + self.bottomLeft()[0] * (1 - y),
            self.topLeft()[1] * y + self.bottomLeft()[1] * (1 - y),
        )

    # Fraction y of way up right side
    def rightAt(self, y):
        return (
            self.topRight()[0] * y + self.bottomRight()[0] * (1 - y),
            self.topRight()[1] * y + self.bottomRight()[1] * (1 - y),
        )

    # Apply x and y offsets in rotated coordinates
    def gOffset(self, point, xoffset=0, yoffset=0):
        return self.gRotate([point[0] + xoffset.point[1] + yoffset], self.rotate, point)

    # Apply a perturbation to positions
    def jitterPos(self):
        if self.jitterGridPoints == 0:
            return 0
        return random.normalvariate(0, self.jitterGridPoints)

    def jitterLW(self):
        if self.jitterLineWidth == 0:
            return 0
        return random.normalvariate(0, self.jitterLineWidth)

    def jitterFS(self):
        if self.jitterFontSize == 0:
            return 0
        return random.normalvariate(0, self.jitterFontSize)

    def jitterFR(self):
        if self.jitterFontRotate == 0:
            return 0
        return random.normalvariate(0, self.jitterFontRotate)

    # Draw bounding box
    def drawBox(self, ax):
        ax.add_line(
            Line2D(
                xdata=(
                    self.gRotate(self.topLeft())[0] + self.jitterPos(),
                    self.gRotate(self.topRight())[0] + self.jitterPos(),
                    self.gRotate(self.bottomRight())[0] + self.jitterPos(),
                    self.gRotate(self.bottomLeft())[0] + self.jitterPos(),
                    self.gRotate(self.topLeft())[0] + self.jitterPos(),
                ),
                ydata=(
                    self.gRotate(self.topLeft())[1] + self.jitterPos(),
                    self.gRotate(self.topRight())[1] + self.jitterPos(),
                    self.gRotate(self.bottomRight())[1] + self.jitterPos(),
                    self.gRotate(self.bottomLeft())[1] + self.jitterPos(),
                    self.gRotate(self.topLeft())[1] + self.jitterPos(),
                ),
                linestyle="solid",
                linewidth=self.linewidth + self.jitterLW(),
                color=self.fgcolour,
                zorder=1,
            )
        )

    # Generate random numbers to fill out the data table
    #  Each month's data is represented by 3 integers (0-9) - x, y, and z
    #  where the accumulated rainfall for that month is x.yz inches.
    def generateNumbers(self):
        self.rdata = []
        for pni in range(3):
            self.rdata.append(random.randint(0, 9))

    # Fill out the table with the random numbers
    def drawNumbers(self, ax):
        (x, y) = self.gCentre()
        inr = self.rdata[0] + self.rdata[1] / 10 + self.rdata[2] / 100
        strv = "%4.2f" % inr
        txp = self.gRotate([x, y])
        ax.text(
            txp[0] + self.jitterPos(),
            txp[1] + self.jitterPos(),
            strv,
            fontdict={
                "family": self.fontFamily,
                "size": self.fontSize + self.jitterFS(),
                "style": self.fontStyle,
                "weight": self.fontWeight,
            },
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1 + self.jitterFR(),
        )
        # Make certain numbers are identical to printed version
        self.rdata[0] = int(strv[0])
        self.rdata[1] = int(strv[2])
        self.rdata[2] = int(strv[3])

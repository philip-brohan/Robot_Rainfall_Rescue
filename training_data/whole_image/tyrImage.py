# Class encapsulating a fake ten-year rainfall data image

import os
import math
import random
import pickle
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


class tyrImage:
    def __init__(self, opdir, docn, **kwargs):
        self.opdir = opdir
        self.docn = docn

        # Parameters defining the image geometry
        self.pageWidth = 640
        self.pageHeight = 1024
        self.xscale = 1.0
        self.yscale = 1.0
        self.xshift = 0.0  # pixels, +ve right
        self.yshift = 0.0  # pixels, +ve up
        self.rotate = 0.0  # degrees clockwise
        self.linewidth = 1.0
        self.bgcolour = (1.0, 1.0, 1.0)
        self.fgcolour = (0.0, 0.0, 0.0)
        self.yearHeight = 0.066  # Fractional height of year row
        self.totalsHeight = 0.105  # Fractional height of totals row
        self.monthsWidth = 0.137  # Fractional width of months row
        self.meansWidth = 0.107  # Fractional width of means row
        self.fontSize = 10
        self.fontFamily = "Arial"
        self.fontStyle = "normal"
        self.fontWeight = "normal"
        self.year = 1941
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
        self.drawGrid(ax_full)
        self.drawFixedText(ax_full)
        self.drawNumbers(ax_full)
        self.drawMeans(ax_full)
        self.drawTotals(ax_full)
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
            0.725 + self.yshift / self.pageHeight,
        )

    def topRight(self):
        return (
            0.96 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.86,
            0.725 + self.yshift / self.pageHeight,
        )

    def bottomLeft(self):
        return (
            0.1 + self.xshift / self.pageWidth,
            0.325 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.4,
        )

    def bottomRight(self):
        return (
            0.96 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.86,
            0.325 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.4,
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

    # Draw grid bounding box
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

    # Draw the grid
    def drawGrid(self, ax):
        lft = self.gRotate(self.leftAt(1.0 - self.yearHeight))
        rgt = self.gRotate(self.rightAt(1.0 - self.yearHeight))
        ax.add_line(
            Line2D(
                xdata=(lft[0] + self.jitterPos(), rgt[0] + self.jitterPos()),
                ydata=(lft[1] + self.jitterPos(), rgt[1] + self.jitterPos()),
                linestyle="solid",
                linewidth=self.linewidth + self.jitterLW(),
                color=self.fgcolour,
                zorder=1,
            )
        )
        lft = self.gRotate(self.leftAt(self.totalsHeight))
        rgt = self.gRotate(self.rightAt(self.totalsHeight))
        ax.add_line(
            Line2D(
                xdata=(lft[0] + self.jitterPos(), rgt[0] + self.jitterPos()),
                ydata=(lft[1] + self.jitterPos(), rgt[1] + self.jitterPos()),
                linestyle="solid",
                linewidth=self.linewidth + self.jitterLW(),
                color=self.fgcolour,
                zorder=1,
            )
        )
        tp = self.gRotate(self.topAt(self.monthsWidth))
        bm = self.gRotate(self.bottomAt(self.monthsWidth))
        ax.add_line(
            Line2D(
                xdata=(tp[0] + self.jitterPos(), bm[0] + self.jitterPos()),
                ydata=(tp[1] + self.jitterPos(), bm[1] + self.jitterPos()),
                linestyle="solid",
                linewidth=self.linewidth,
                color=self.fgcolour,
                zorder=1,
            )
        )
        tp = self.gRotate(self.topAt(1.0 - self.meansWidth))
        bm = self.gRotate(self.bottomAt(1.0 - self.meansWidth))
        ax.add_line(
            Line2D(
                xdata=(tp[0] + self.jitterPos(), bm[0] + self.jitterPos()),
                ydata=(tp[1] + self.jitterPos(), bm[1] + self.jitterPos()),
                linestyle="solid",
                linewidth=self.linewidth + self.jitterLW(),
                color=self.fgcolour,
                zorder=1,
            )
        )
        for yrl in range(1, 10):
            x = self.monthsWidth + yrl * (1.0 - self.meansWidth - self.monthsWidth) / 10
            tp = self.gRotate(self.topAt(x))
            bm = self.gRotate(self.bottomAt(x))
            ax.add_line(
                Line2D(
                    xdata=(tp[0] + self.jitterPos(), bm[0] + self.jitterPos()),
                    ydata=(tp[1] + self.jitterPos(), bm[1] + self.jitterPos()),
                    linestyle="solid",
                    linewidth=self.linewidth + self.jitterLW(),
                    color=self.fgcolour,
                    zorder=1,
                )
            )

    # Add the fixed text
    def drawFixedText(self, ax):
        tp = self.topAt(self.monthsWidth / 2)
        lft = self.leftAt(1.0 - self.yearHeight / 2)
        txp = self.gRotate([tp[0], lft[1]])
        ax.text(
            txp[0],
            txp[1],
            "Year",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1,
        )
        tp = self.topAt(1.0 - self.meansWidth / 2)
        lft = self.leftAt(1.0 - self.yearHeight / 2)
        txp = self.gRotate([tp[0], lft[1]])
        ax.text(
            txp[0],
            txp[1],
            "Means",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1,
        )
        tp = self.topAt(self.monthsWidth / 2)
        lft = self.leftAt(self.totalsHeight / 2)
        txp = self.gRotate([tp[0], lft[1]])
        ax.text(
            txp[0],
            txp[1],
            "Totals",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1,
        )
        months = (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
        tp = self.topAt(self.monthsWidth / 10)
        for mdx in range(len(months)):
            lft = self.leftAt(
                1.0
                - self.yearHeight
                - (mdx + 1)
                * (1.0 - self.yearHeight - self.totalsHeight)
                / (len(months) + 1)
            )
            txp = self.gRotate([tp[0], lft[1]])
            ax.text(
                txp[0],
                txp[1],
                months[mdx],
                fontsize=8,
                horizontalalignment="left",
                verticalalignment="center",
                rotation=self.rotate * -1,
                rotation_mode="anchor",
            )
        lft = self.leftAt(1.0 - self.yearHeight / 2)
        for ydx in range(10):
            x = (
                self.monthsWidth
                + (ydx + 0.5) * (1.0 - self.meansWidth - self.monthsWidth) / 10
            )
            tp = self.topAt(x)
            txp = self.gRotate([tp[0], lft[1]])
            ax.text(
                txp[0],
                txp[1],
                "%04d" % (self.year + ydx),
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
                rotation=self.rotate * -1,
            )

    # Generate random numbers to fill out the data table
    #  Each month's data is represented by 3 integers (0-9) - x, y, and z
    #  where the accumulated rainfall for that month is x.yz inches.
    def generateNumbers(self):
        self.rdata = []
        for yri in range(10):
            ydata = []
            for mni in range(12):
                mdata = []
                for pni in range(3):
                    mdata.append(random.randint(0, 9))
                ydata.append(mdata)
            self.rdata.append(ydata)

    # Fill out the table with the random numbers
    def drawNumbers(self, ax):
        for yri in range(10):
            x = (
                self.monthsWidth
                + (yri + 0.5) * (1.0 - self.meansWidth - self.monthsWidth) / 10
            )
            tp = self.topAt(x)
            for mni in range(12):
                lft = self.leftAt(
                    1.0
                    - self.yearHeight
                    - (mni + 1) * (1.0 - self.yearHeight - self.totalsHeight) / (12 + 1)
                )
                inr = (
                    self.rdata[yri][mni][0]
                    + self.rdata[yri][mni][1] / 10
                    + self.rdata[yri][mni][2] / 100
                )
                strv = "%4.2f" % inr
                txp = self.gRotate([tp[0], lft[1]])
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
                self.rdata[yri][mni][0] = int(strv[0])
                self.rdata[yri][mni][1] = int(strv[2])
                self.rdata[yri][mni][2] = int(strv[3])

    # Add the monthly means
    def drawMeans(self, ax):
        tp = self.topAt(1.0 - self.meansWidth / 2)
        self.rdata.append([])
        for mni in range(12):
            lft = self.leftAt(
                1.0
                - self.yearHeight
                - (mni + 1) * (1.0 - self.yearHeight - self.totalsHeight) / (12 + 1)
            )
            inr = 0.0
            for yri in range(10):
                inr += (
                    self.rdata[yri][mni][0]
                    + self.rdata[yri][mni][1] / 10
                    + self.rdata[yri][mni][2] / 100
                )
            inr /= 10
            strv = "%04.2f" % inr
            txp = self.gRotate([tp[0], lft[1]])
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
            mm = []
            mm.append(int(strv[0]))
            mm.append(int(strv[2]))
            mm.append(int(strv[3]))
            self.rdata[10].append(mm)

    # Add the annual totals
    def drawTotals(self, ax):
        lft = self.leftAt(self.totalsHeight / 2)
        self.rdata.append([])
        for yri in range(10):
            x = (
                self.monthsWidth
                + (yri + 0.5) * (1.0 - self.meansWidth - self.monthsWidth) / 10
            )
            tp = self.topAt(x)
            inr = 0.0
            for mni in range(12):
                inr += (
                    self.rdata[yri][mni][0]
                    + self.rdata[yri][mni][1] / 10
                    + self.rdata[yri][mni][2] / 100
                )
            if inr > 99:
                inr = inr % 100
            strv = "%05.2f" % inr
            txp = self.gRotate([tp[0], lft[1]])
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
            mm = []
            mm.append(int(strv[0]))
            mm.append(int(strv[2]))
            mm.append(int(strv[3]))
            self.rdata[10].append(mm)

    # Add the annual totals
    def drawTotals(self, ax):
        lft = self.leftAt(self.totalsHeight / 2)
        self.rdata.append([])
        for yri in range(10):
            x = (
                self.monthsWidth
                + (yri + 0.5) * (1.0 - self.meansWidth - self.monthsWidth) / 10
            )
            tp = self.topAt(x)
            inr = 0.0
            for mni in range(12):
                inr += (
                    self.rdata[yri][mni][0]
                    + self.rdata[yri][mni][1] / 10
                    + self.rdata[yri][mni][2] / 100
                )
            if inr > 99:
                inr = inr % 100
            strv = "%05.2f" % inr
            txp = self.gRotate([tp[0], lft[1]])
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
            at = []
            at.append(int(strv[0]))
            at.append(int(strv[1]))
            at.append(int(strv[3]))
            at.append(int(strv[4]))
            self.rdata[11].append(at)

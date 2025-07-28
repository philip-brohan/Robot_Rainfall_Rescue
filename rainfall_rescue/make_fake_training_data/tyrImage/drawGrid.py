# Functions to draw the central table

from matplotlib.lines import Line2D
from rainfall_rescue.make_fake_training_data.tyrImage.constants import months
from rainfall_rescue.make_fake_training_data.tyrImage.geometry import (
    gRotate,
    topAt,
    bottomAt,
    leftAt,
    rightAt,
    topLeft,
    topRight,
    bottomLeft,
    bottomRight,
)
from rainfall_rescue.make_fake_training_data.tyrImage.jitter import jitterPos, jitterLW


# Draw grid bounding box
def drawBox(self, ax):
    ax.add_line(
        Line2D(
            xdata=(
                gRotate(self, topLeft(self))[0] + jitterPos(self),
                gRotate(self, topRight(self))[0] + jitterPos(self),
                gRotate(self, bottomRight(self))[0] + jitterPos(self),
                gRotate(self, bottomLeft(self))[0] + jitterPos(self),
                gRotate(self, topLeft(self))[0] + jitterPos(self),
            ),
            ydata=(
                gRotate(self, topLeft(self))[1] + jitterPos(self),
                gRotate(self, topRight(self))[1] + jitterPos(self),
                gRotate(self, bottomRight(self))[1] + jitterPos(self),
                gRotate(self, bottomLeft(self))[1] + jitterPos(self),
                gRotate(self, topLeft(self))[1] + jitterPos(self),
            ),
            linestyle="solid",
            linewidth=self.linewidth + jitterLW(self),
            color=self.fgcolour,
            zorder=1,
        )
    )


# Draw the grid
def drawGrid(self, ax):
    # Horizontal lines above the years column
    for idx_h in range(3):
        lft = gRotate(self, leftAt(self, 1.0 - self.headerHeight * (idx_h + 1)))
        rgt = gRotate(self, rightAt(self, 1.0 - self.headerHeight * (idx_h + 1)))
        ax.add_line(
            Line2D(
                xdata=(lft[0] + jitterPos(self), rgt[0] + jitterPos(self)),
                ydata=(lft[1] + jitterPos(self), rgt[1] + jitterPos(self)),
                linestyle="solid",
                linewidth=self.linewidth + jitterLW(self),
                color=self.fgcolour,
                zorder=1,
            )
        )
    # Horizontal line under the years column
    lft = gRotate(self, leftAt(self, 1.0 - self.yearHeight - self.headerHeight * 3))
    rgt = gRotate(self, rightAt(self, 1.0 - self.yearHeight - self.headerHeight * 3))
    ax.add_line(
        Line2D(
            xdata=(lft[0] + jitterPos(self), rgt[0] + jitterPos(self)),
            ydata=(lft[1] + jitterPos(self), rgt[1] + jitterPos(self)),
            linestyle="solid",
            linewidth=self.linewidth + jitterLW(self),
            color=self.fgcolour,
            zorder=1,
        )
    )
    # Horizontal line over the totals row
    lft = gRotate(self, leftAt(self, self.totalsHeight))
    rgt = gRotate(self, rightAt(self, self.totalsHeight))
    ax.add_line(
        Line2D(
            xdata=(lft[0] + jitterPos(self), rgt[0] + jitterPos(self)),
            ydata=(lft[1] + jitterPos(self), rgt[1] + jitterPos(self)),
            linestyle="solid",
            linewidth=self.linewidth + jitterLW(self),
            color=self.fgcolour,
            zorder=1,
        )
    )
    # Vertical lines marking the month labels column
    tp = gRotate(self, topAt(self, self.monthsWidth))
    bm = gRotate(self, bottomAt(self, self.monthsWidth))
    ax.add_line(
        Line2D(
            xdata=(tp[0] + jitterPos(self), bm[0] + jitterPos(self)),
            ydata=(tp[1] + jitterPos(self), bm[1] + jitterPos(self)),
            linestyle="solid",
            linewidth=self.linewidth,
            color=self.fgcolour,
            zorder=1,
        )
    )
    # Vertical lines marking the means column
    tp = gRotate(self, topAt(self, 1.0 - self.meansWidth))
    bm = gRotate(self, bottomAt(self, 1.0 - self.meansWidth))
    ax.add_line(
        Line2D(
            xdata=(tp[0] + jitterPos(self), bm[0] + jitterPos(self)),
            ydata=(tp[1] + jitterPos(self), bm[1] + jitterPos(self)),
            linestyle="solid",
            linewidth=self.linewidth + jitterLW(self),
            color=self.fgcolour,
            zorder=1,
        )
    )
    # Vertical lines marking each of the 10 years
    for yrl in range(1, 10):
        x = self.monthsWidth + yrl * (1.0 - self.meansWidth - self.monthsWidth) / 10
        tp = gRotate(self, topAt(self, x))
        bm = gRotate(self, bottomAt(self, x))
        ax.add_line(
            Line2D(
                xdata=(tp[0] + jitterPos(self), bm[0] + jitterPos(self)),
                ydata=(tp[1] + jitterPos(self), bm[1] + jitterPos(self)),
                linestyle="solid",
                linewidth=self.linewidth + jitterLW(self),
                color=self.fgcolour,
                zorder=1,
            )
        )


# Add the fixed text
def drawFixedText(self, ax):
    tp = topAt(self, self.monthsWidth / 2)
    lft = leftAt(self, 1.0 - self.yearHeight / 2 - self.headerHeight * 3)
    txp = gRotate(self, [tp[0], lft[1]])
    ax.text(
        txp[0],
        txp[1],
        "Year",
        fontsize=18,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=self.rotate * -1,
    )
    tp = topAt(self, 1.0 - self.meansWidth / 2)
    lft = leftAt(self, 1.0 - self.yearHeight / 2 - self.headerHeight * 3)
    txp = gRotate(self, [tp[0], lft[1]])
    ax.text(
        txp[0],
        txp[1],
        "Means",
        fontsize=18,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=self.rotate * -1,
    )
    tp = topAt(self, self.monthsWidth / 2)
    lft = leftAt(self, self.totalsHeight / 2)
    txp = gRotate(self, [tp[0], lft[1]])
    ax.text(
        txp[0],
        txp[1],
        "Totals",
        fontsize=18,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=self.rotate * -1,
    )
    tp = topAt(self, self.monthsWidth / 10)
    for mdx in range(len(months)):
        lft = leftAt(
            self,
            1.0
            - self.yearHeight
            - self.headerHeight * 3
            - (mdx + 1)
            * (1.0 - self.yearHeight - self.headerHeight * 3 - self.totalsHeight)
            / (len(months) + 1),
        )
        txp = gRotate(self, [tp[0], lft[1]])
        ax.text(
            txp[0],
            txp[1],
            months[mdx],
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
            rotation=self.rotate * -1,
            rotation_mode="anchor",
        )
    lft = leftAt(self, 1.0 - self.yearHeight / 2 - self.headerHeight * 3)
    for ydx in range(10):
        x = (
            self.monthsWidth
            + (ydx + 0.5) * (1.0 - self.meansWidth - self.monthsWidth) / 10
        )
        tp = topAt(self, x)
        txp = gRotate(self, [tp[0], lft[1]])
        ax.text(
            txp[0],
            txp[1],
            "%04d" % (self.meta["Year"] + ydx),
            fontsize=18,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1,
        )

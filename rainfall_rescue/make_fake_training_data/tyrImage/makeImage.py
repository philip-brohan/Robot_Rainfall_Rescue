# tyrImage instance function to draw the image

import os
import matplotlib
from matplotlib.figure import Figure
from rainfall_rescue.make_fake_training_data.tyrImage.drawGrid import (
    drawBox,
    drawGrid,
    drawFixedText,
)
from rainfall_rescue.make_fake_training_data.tyrImage.metadata import (
    drawStationNumber,
    drawStationName,
    drawFakeMetadata,
)
from rainfall_rescue.make_fake_training_data.tyrImage.numbers import (
    drawNumbers,
    drawMeans,
    drawTotals,
)


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
    ax_full = fig.add_axes([0, 0, 1, 1])
    ax_full.set_xlim([0, 1])
    ax_full.set_ylim([0, 1])
    ax_full.set_axis_off()

    # Paint the background
    ax_full.add_patch(
        matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor=self.bgcolour)
    )

    # Draw the figure
    drawBox(self, ax_full)
    drawGrid(self, ax_full)
    drawFixedText(self, ax_full)
    drawNumbers(self, ax_full)
    drawMeans(self, ax_full)
    drawTotals(self, ax_full)
    drawStationNumber(self, ax_full)
    drawStationName(self, ax_full)
    drawFakeMetadata(self, ax_full)
    if not os.path.isdir("%s/images" % self.opdir):
        os.makedirs("%s/images" % self.opdir)
    fig.savefig("%s/images/%s.png" % (self.opdir, self.docn))

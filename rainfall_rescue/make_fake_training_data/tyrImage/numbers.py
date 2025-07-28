# Generate and draw the random nubers used as fake rainfall
import random

from rainfall_rescue.make_fake_training_data.tyrImage.constants import months
from rainfall_rescue.make_fake_training_data.tyrImage.jitter import (
    jitterPos,
    jitterFS,
    jitterFR,
)
from rainfall_rescue.make_fake_training_data.tyrImage.geometry import (
    topAt,
    leftAt,
    gRotate,
)


def isPresent(probability):
    """Return True with a given probability."""
    return random.random() < probability


# Random rainfall numbers arranged as a dictionary
def generateNumbers(self):
    ydata = {}
    for mni in range(12):
        ydata[months[mni]] = ["null"] * 10
    # 10 years of monthly observations
    isnull = True
    for yri in range(10):
        if isPresent(self.Yearprobability):
            isnull = False
        for mni in range(12):
            mdata = []
            for pni in range(3):
                mdata.append(random.randint(0, 9))
            if not isnull and isPresent(self.Dataprobability):
                ydata[months[mni]][yri] = "%d.%d%d" % tuple(
                    mdata
                )  # not missing for this month
        isnull = True
    # Add the totals
    ydata["Totals"] = ["null"] * 10
    if isPresent(self.Totalsprobability):
        for yri in range(10):
            total = 0
            for mni in range(12):
                if ydata[months[mni]][yri] != "null":
                    total += float(ydata[months[mni]][yri])
            if total > 0:
                ydata["Totals"][yri] = "%.2f" % total
    # Add the means
    ydata["Means"] = ["null"] * 12
    if isPresent(self.Meansprobability):
        for mni in range(12):
            total = 0
            count = 0
            for yri in range(10):
                if ydata[months[mni]][yri] != "null":
                    total += float(ydata[months[mni]][yri])
                    count += 1
            if count > 0:
                ydata["Means"][mni] = "%.2f" % (total / count)
    return ydata


# Fill out the table with the random numbers
def drawNumbers(self, ax):
    for yri in range(10):
        x = (
            self.monthsWidth
            + (yri + 0.5) * (1.0 - self.meansWidth - self.monthsWidth) / 10
        )
        tp = topAt(self, x)
        for mni in range(12):
            lft = leftAt(
                self,
                1.0
                - self.yearHeight
                - self.headerHeight * 3
                - (mni + 1)
                * (1.0 - self.yearHeight - self.headerHeight * 3 - self.totalsHeight)
                / (12 + 1),
            )
            strv = self.rdata[months[mni]][yri]
            txp = gRotate(self, [tp[0], lft[1]])
            if strv != "null":
                ax.text(
                    txp[0] + jitterPos(self),
                    txp[1] + jitterPos(self),
                    strv,
                    fontdict={
                        "family": self.fontFamily,
                        "size": self.fontSize + jitterFS(self),
                        "style": self.fontStyle,
                        "weight": self.fontWeight,
                    },
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=self.rotate * -1 + jitterFR(self),
                )


# Add the monthly means
def drawMeans(self, ax):
    tp = topAt(self, 1.0 - self.meansWidth / 2)
    for mni in range(12):
        lft = leftAt(
            self,
            1.0
            - self.yearHeight
            - self.headerHeight * 3
            - (mni + 1)
            * (1.0 - self.yearHeight - self.headerHeight * 3 - self.totalsHeight)
            / (12 + 1),
        )
        strv = self.rdata["Means"][mni]
        if strv != "null":
            txp = gRotate(self, [tp[0], lft[1]])
            ax.text(
                txp[0] + jitterPos(self),
                txp[1] + jitterPos(self),
                strv,
                fontdict={
                    "family": self.fontFamily,
                    "size": self.fontSize + jitterFS(self),
                    "style": self.fontStyle,
                    "weight": self.fontWeight,
                },
                horizontalalignment="center",
                verticalalignment="center",
                rotation=self.rotate * -1 + jitterFR(self),
            )


# Add the annual totals
def drawTotals(self, ax):
    lft = leftAt(self, self.totalsHeight / 2)
    for yri in range(10):
        x = (
            self.monthsWidth
            + (yri + 0.5) * (1.0 - self.meansWidth - self.monthsWidth) / 10
        )
        tp = topAt(self, x)
        strv = self.rdata["Totals"][yri]
        if strv != "null":
            txp = gRotate(self, [tp[0], lft[1]])
            ax.text(
                txp[0] + jitterPos(self),
                txp[1] + jitterPos(self),
                strv,
                fontdict={
                    "family": self.fontFamily,
                    "size": self.fontSize + jitterFS(self),
                    "style": self.fontStyle,
                    "weight": self.fontWeight,
                },
                horizontalalignment="center",
                verticalalignment="center",
                rotation=self.rotate * -1 + jitterFR(self),
            )

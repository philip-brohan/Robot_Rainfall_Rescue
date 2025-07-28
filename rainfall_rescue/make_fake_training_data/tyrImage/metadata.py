# Functions to add the station metadata

import random
from rainfall_rescue.make_fake_training_data.tyrImage.jitter import (
    jitterFS,
    jitterTP,
)
from rainfall_rescue.make_fake_training_data.tyrImage.geometry import (
    gRotate,
)


def isPresent(probability):
    """Return True with a given probability."""
    return random.random() < probability


def randomstring(length=10):
    """Generate a random string of fixed length."""
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    return "".join(random.choice(letters) for i in range(length))


# Create the station metadata
def generateStationMetadata(self):
    meta = {}
    meta["Year"] = random.randint(185, 200) * 10 + 1
    if isPresent(self.stationNumberprobability):
        meta["stationNumber"] = random.randint(0, 10 ** random.randint(1, 4))
    else:
        meta["stationNumber"] = "null"
    if isPresent(self.stationNameprobability):
        meta["stationName"] = randomstring(random.randint(5, 20))
    else:
        meta["stationName"] = "null"
    return meta


# Add the station number (top right corner)
def drawStationNumber(self, ax):
    tp = 0.93 + jitterTP(self)
    lft = 0.95 + jitterTP(self)
    txp = gRotate(self, [tp, lft])
    if self.meta["stationNumber"] != "null":
        ax.text(
            txp[0],
            txp[1],
            "%d" % self.meta["stationNumber"],
            fontdict={
                "family": self.fontFamily,
                "size": self.fontSize + jitterFS(self),
                "style": self.fontStyle,
                "weight": self.fontWeight,
            },
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1,
        )


# Add the station Name
def drawStationName(self, ax):
    tp = 0.5 + jitterTP(self)
    lft = 0.92 + jitterTP(self)
    txp = gRotate(self, [tp, lft])
    if self.meta["stationName"] != "null":
        ax.text(
            txp[0],
            txp[1],
            "RAINFALL at  %s" % self.meta["stationName"],
            fontdict={
                "family": self.fontFamily,
                "size": self.fontSize + jitterFS(self),
                "style": self.fontStyle,
                "weight": self.fontWeight,
            },
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1,
        )


# Add a single piece of fake metadata - stuff to be ignored
def drawFakeMetadatum(self, ax, tp, lft, text):
    if isPresent(self.fakeMetaprobability):
        txp = gRotate(self, [tp, lft])
        ax.text(
            txp[0],
            txp[1],
            "%s" % text,
            fontdict={
                "family": self.fontFamily,
                "size": self.fontSize + jitterFS(self),
                "style": self.fontStyle,
                "weight": self.fontWeight,
            },
            horizontalalignment="center",
            verticalalignment="center",
            rotation=self.rotate * -1,
        )


# Add a few pieces of fake metadata
def drawFakeMetadata(self, ax):
    drawFakeMetadatum(
        self,
        ax,
        0.25 + jitterTP(self),
        0.85 + jitterTP(self),
        "County of  %s" % randomstring(10),
    )
    drawFakeMetadatum(
        self,
        ax,
        0.75 + jitterTP(self),
        0.85 + jitterTP(self),
        "River Basin: %s" % randomstring(19),
    )
    drawFakeMetadatum(
        self,
        ax,
        0.25 + jitterTP(self),
        0.80 + jitterTP(self),
        "Type of Gauge: %s" % randomstring(8),
    )
    drawFakeMetadatum(
        self,
        ax,
        0.75 + jitterTP(self),
        0.80 + jitterTP(self),
        "Observer: %s" % randomstring(20),
    )

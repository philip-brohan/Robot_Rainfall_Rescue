# Class encapsulating a fake ten-year rainfall data image

import random

from rainfall_rescue.make_fake_training_data.tyrImage.makeImage import makeImage
from rainfall_rescue.make_fake_training_data.tyrImage.makeCSV import makeCSV
from rainfall_rescue.make_fake_training_data.tyrImage.constants import available_fonts
from rainfall_rescue.make_fake_training_data.tyrImage.numbers import (
    generateNumbers,
)
from rainfall_rescue.make_fake_training_data.tyrImage.metadata import (
    generateStationMetadata,
)


class tyrImage:
    def __init__(self, opdir, docn, **kwargs):
        self.opdir = opdir
        self.docn = docn

        # Parameters defining the image geometry
        self.pageWidth = 1180 + random.normalvariate(0, 50)  # pixels, +ve right
        self.pageHeight = 1900 + random.normalvariate(0, 50)  # pixels, +ve up
        self.xscale = 1.0
        self.yscale = 1.0
        self.xshift = random.normalvariate(0, 10)  # pixels, +ve right
        self.yshift = random.normalvariate(0, 20)  # pixels, +ve up
        self.rotate = random.normalvariate(0, 0.5)  # degrees clockwise
        self.linewidth = random.uniform(0.5, 1.0)  # Line width in points
        self.bgcolour = (0.9, 0.9, 0.9)
        self.fgcolour = (0.0, 0.0, 0.0)
        self.headerHeight = 0.059  # Fractional height rows above years
        self.yearHeight = 0.059  # Fractional height of year row
        self.totalsHeight = 0.105  # Fractional height of totals row
        self.monthsWidth = 0.137  # Fractional width of months row
        self.meansWidth = 0.107  # Fractional width of means row
        self.fontSize = 18
        self.fontFamily = random.choice(available_fonts)
        self.fontStyle = "normal"
        self.fontWeight = "normal"
        # Noise parameters
        self.jitterFontSize = 0.5
        self.jitterFontRotate = 1.0
        self.jitterGridPoints = 0.001
        self.jitterLineWidth = 0.1
        self.jitterTextPosition = 0.005
        # Probability data is present
        self.fakeMetaprobability = 0.9
        self.stationNumberprobability = 0.9
        self.stationNameprobability = 0.9
        self.Meansprobability = 0.5
        self.Totalsprobability = 0.95
        self.Yearprobability = 0.95
        self.Dataprobability = 0.99

        self.meta = generateStationMetadata(self)
        self.rdata = generateNumbers(self)

        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    setattr(self, key, value)
            else:
                raise ValueError("No parameter %s" % key)

        makeImage(self)
        makeCSV(self)

# functious to make random perturbations to image features

import random


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


def jitterTP(self):
    if self.jitterTextPosition == 0:
        return 0
    return random.normalvariate(0, self.jitterTextPosition)

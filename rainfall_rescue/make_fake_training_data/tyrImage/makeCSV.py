# Instance function for the tyrImage module

import os
from rainfall_rescue.make_fake_training_data.tyrImage.constants import months


# Convert the data into a CSV string
# String format is analagous to that taken from the original data
def makeCSV(self):
    dict = {}
    dict["Name"] = self.meta["stationName"]
    dict["Number"] = self.meta["stationNumber"]
    dict["Years"] = [str(self.meta["Year"] + i) for i in range(10)]
    for imnth in range(12):
        dict[months[imnth]] = self.rdata[months[imnth]]
    dict["Totals"] = self.rdata["Totals"]
    if not os.path.isdir("%s/csv" % self.opdir):
        os.makedirs("%s/csv" % self.opdir)
    with open("%s/csv/%s.csv" % (self.opdir, self.docn), "w") as pf:
        pf.write(str(dict))

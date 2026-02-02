#!/usr/bin/env python

# Take all the RR csv files and insert them as Milvus vectors.
import os
import sys
from pymilvus import MilvusClient
from rainfall_rescue.utils.pairs import get_index_list, load_pair

# Mhere to put the db file
db_file = f"{os.getenv('PDIR')}/RR_monthly.db"

# Set up a Milvus client
client = MilvusClient(uri=db_file)

# Get a list of all the RR pages
pages = get_index_list(fake=False, shuffle=False)


# Add a single annual vector to the db
def insert_year(client, station_number, station_name, year, monthly_averages):
    client.insert(
        "rainfall_rescue",
        {
            "monthly_averages": monthly_averages,
            "station_number": station_number,
            "station_name": station_name,
            "year": year,
        },
    )


monthNumbers = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


# Loop over all the pages
count = 0
for p in pages:
    img, csv = load_pair(p)
    try:
        station_number = csv["Number"]
    except KeyError:
        print("No station number for page:", p)
        station_number = "UNKNOWN"
        continue
    try:
        station_name = csv["Name"]
    except KeyError:
        print("No station name for page:", p)
        station_name = "UNKNOWN"
    try:
        years = csv["Years"]
    except KeyError:
        print("No years for page:", p)
        continue
    # Loop over all the years for this station
    for idx in range(10):
        year = csv["Years"][idx]
        monthly_averages = [0] * 12
        for month in monthNumbers.keys():
            try:
                value = float(csv[month][idx])
            except ValueError:
                # print("Bad value:", csv[month][idx], "for", station_number, year, month)
                value = 0.0
            monthly_averages[monthNumbers[month] - 1] = value
        insert_year(client, station_number, station_name, year, monthly_averages)
        count += 1
        if count % 1000 == 0:
            print(f"Inserted {count} vectors so far")
    print(f"Inserted station {station_number} year {years[0]}")
print(f"Finished inserting {count} vectors")

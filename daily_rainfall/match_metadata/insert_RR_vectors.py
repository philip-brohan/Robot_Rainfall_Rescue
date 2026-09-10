#!/usr/bin/env python

# Take all the RR csv files and insert them as Milvus vectors.
import os
import sys
from pymilvus import MilvusClient
from daily_rainfall.match_metadata.mdpairs import find_csv_files, load_station_csv

# Mhere to put the db file
db_file = f"{os.getenv('PDIR')}/RR_monthly.db"

# Set up a Milvus client
client = MilvusClient(uri=db_file)

# Get a list of all the RR pages = find_csv_files()
files = find_csv_files()

# Speed things up - only interested in a limited range of years
startyear = 1871
endyear = 1880


# Add a single annual vector to the db
def insert_year(
    client, station_number, station_name, latitude, longitude, year, monthly_averages
):
    client.insert(
        "rainfall_rescue",
        {
            "monthly_averages": monthly_averages,
            "station_number": station_number,
            "station_name": station_name,
            "Longitude": longitude,
            "Latitude": latitude,
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


# Loop over all the station records
count = 0
for p in files:
    csv = load_station_csv(p)
    if csv["Latitude"] == "null" or csv["Longitude"] == "null":
        print("No lat/lon for page:", p)
        continue
    try:
        station_number = csv["Number"]
    except KeyError:
        print("No station number for page:", p)
        station_number = "UNKNOWN"
        continue
    try:
        station_name = csv["Name"]
        print(f"Processing station {station_number} - {station_name}")
    except KeyError:
        print("No station name for page:", p)
        station_name = "UNKNOWN"
    try:
        years = csv["Years"]
    except KeyError:
        print("No years for page:", p)
        continue
    # Loop over all the years for this station
    for idx in range(len(years)):
        year = years[idx]
        if year == "null" or int(year) < startyear or int(year) > endyear:
            continue
        monthly_averages = [0] * 12
        for month in monthNumbers.keys():
            try:
                value = float(csv[month][idx])
            except ValueError:
                # print("Bad value:", csv[month][idx], "for", station_number, year, month)
                value = 0.0
            except KeyError as e:
                print(csv.keys())
                raise e
            monthly_averages[monthNumbers[month] - 1] = value
        insert_year(
            client,
            station_number,
            station_name,
            csv["Latitude"],
            csv["Longitude"],
            year,
            monthly_averages,
        )
        count += 1
        if count % 1000 == 0:
            print(f"Inserted {count} vectors so far")
    print(f"Inserted station {station_number} year {years[0]}")
print(f"Finished inserting {count} vectors")

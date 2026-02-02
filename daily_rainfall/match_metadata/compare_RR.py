# Functions for comparing monthly averages with the RR data.

import os
import sys
from pymilvus import MilvusClient

# Location of the vector DB with the rainfall rescue monthly averages
db_file = f"{os.getenv('PDIR')}/RR_monthly.db"


# Get a client to access the db
def get_RR_monthly_db():

    # Set up a Milvus client
    client = MilvusClient(uri=db_file)
    return client


# Search the db for the closest match to a given monthly average vector
def search_RR_monthly_db(client, monthly_averages, top_k=1):
    results = client.search(
        collection_name="rainfall_rescue",
        data=[monthly_averages],
        anns_field="monthly_averages",
        limit=top_k,
        output_fields=["station_number", "station_name", "year", "monthly_averages"],
    )
    return results

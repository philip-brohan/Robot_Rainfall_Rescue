#!/usr/bin/env python

# Make a Milvus vector db to store the monthly averages from RR
# Then we can find the monthly record that corresponds to a daily
#  record by similarity search on the vector of monthly averages.

import os
from pymilvus import MilvusClient

# Mhere to put the db file
db_file = f"{os.getenv('PDIR')}/RR_monthly.db"

# Set up a Milvus client
client = MilvusClient(uri=db_file)
# Create a collection in quick setup mode
if client.has_collection(collection_name="rainfall_rescue"):
    client.drop_collection(collection_name="rainfall_rescue")
client.create_collection(
    collection_name="rainfall_rescue",
    vector_field_name="monthly_averages",
    dimension=12,
    auto_id=True,
    enable_dynamic_field=True,
    metric_type="L2",
)

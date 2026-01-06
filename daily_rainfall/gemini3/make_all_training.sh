#!/bin/bash

# Convert the full list of gemini3 page outputs into training format

while IFS= read -r line; do
  echo "./gemini_to_training_json.py --record_id=$line"
  ./gemini_to_training_json.py "--record_id=$line"
done < for_training.txt


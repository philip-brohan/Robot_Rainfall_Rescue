#!/bin/bash

# Convert the selected of gemini3 page outputs into training format

while IFS= read -r line; do
  echo "./gemini_to_training_json.py --image=$line --op_group=validation"
  ./gemini_to_training_json.py --op_group=validation --image=$line
done < for_validation.txt


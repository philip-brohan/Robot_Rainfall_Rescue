#!/bin/bash

# Should only need to run this once.

# PIM into the Member role of the az-rg-Gen2SC-AI4-Climate-Role-Engineer group before running this script

az login --use-device-code
az ml workspace update --resource-group $AZML_RESOURCE_GROUP --name $AZML_WORKSPACE_NAME --image-build-compute Basic

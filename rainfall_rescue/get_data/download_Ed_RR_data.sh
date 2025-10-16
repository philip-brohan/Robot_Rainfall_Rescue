#!/bin/bash

# Get Ed Hawkins' Rainfall Rescue github repository
#  (contains all the images and csv files with transcribed data).
#  Note - 17Gb zip file, 19Gb unpacked

wget -O $SCRATCH/rainfall-rescue.zip https://github.com/ed-hawkins/rainfall-rescue/archive/master.zip
unzip -d $SCRATCH $SCRATCH/rainfall-rescue.zip

# Reset all the access times (or $SCRATCH will delete them as too old).
find $SCRATCH/rainfall-rescue-master -type f -exec touch {} +

# Copy the images from Ed's GDrive (shared with me).
rclone copy gdrive:IMAGES --drive-shared-with-me /data/scratch/philip.brohan/rainfall-rescue-master//IMAGES

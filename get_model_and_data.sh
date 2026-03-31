#!/usr/bin/env bash

# Usage:
# ./download_gdrive_folder.sh FOLDER_ID [OUTPUT_DIR]

set -e

FOLDER_ID="14b623LKcaOMFc9OavPLrzPyQwo6p30uW"
OUTPUT_DIR="."

echo "Downloading folder $FOLDER_ID..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$OUTPUT_DIR"

echo "Download complete. Unpacking."

echo "Input password for models.zip"
unzip models.zip
if [ -d models ]
then
    rm -fr models.zip
fi

echo "Input password for data.zip"
unzip  data.zip
if [ -d data ]
then
    rm -fr data.zip
fi

echo "Input password for pickles.zip"
unzip pickles.zip

if [ -d pickles ]
then   
    mv pickles notebooks
    rm -fr pickles.zip
fi

echo "Done."

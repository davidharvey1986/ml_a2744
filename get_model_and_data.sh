#!/usr/bin/env bash

# Usage:
# ./download_gdrive_folder.sh FOLDER_ID [OUTPUT_DIR]

set -e

FOLDER_ID="14b623LKcaOMFc9OavPLrzPyQwo6p30uW"
OUTPUT_DIR="."

echo "Downloading folder $FOLDER_ID..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$OUTPUT_DIR"

echo "Download complete. Unpacking."
tar -xvf models.tar.gz && rm -fr models.tar.gz
tar -xvf data.tar.gz && rm -fr data.tar.gz
tar -xvf pickles.tar.gz && mv pickles notebooks && rm -fr pickles.tar.gz
echo "Done."

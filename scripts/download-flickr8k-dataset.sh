#!/bin/bash

FLICKR_DIR="../flickr8k_data"

echo "Creating directory: ${FLICKR_DIR}"
mkdir -p $FLICKR_DIR
cd $FLICKR_DIR

echo "Downloading Flickr8K data set (go grab a coffee)..."

wget -nc http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip
wget -nc http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip

echo "Unzipping Flickr8K data set..."
unzip -n Flickr8k_text.zip -d captions -x '__MACOSX/*'
unzip -j -n Flickr8k_Dataset.zip 'Flicker8k_Dataset/*' -d images
# Image Captioning

## Image Encoding

Images are encoded as a vector of length 4096 using 2nd to last layer of the
VGG16 image-classification model.

```
$ python encode.py -i spurs.jpg

[[ 0.          0.2089304   1.39936841 ...,  0.          0.          0.85876471]]
(1, 4096)
```

## Download Flickr8K Data Set

Bash script to download Flickr8K images and their captions. Creates two folders: `/flickr8k_data/images/` and `/flickr8k_data/captions/`.

```
cd scripts
./download-flickr8k-dataset.sh
```
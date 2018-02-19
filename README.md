# Image Captioning

## Image Encoding

Images are encoded as a vector of length 4096 using 2nd to last layer of the
VGG16 image-classification model.

![COYS](spurs.jpg)

```
$ python encode.py -i spurs.jpg

[[ 0.          0.2089304   1.39936841 ...,  0.          0.          0.85876471]]
(1, 4096)
```

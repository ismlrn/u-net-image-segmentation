# U-Net Image Segmantation
A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation.
Image segmentation is the division of an image into regions or categories, which
correspond to different objects or parts of objects. After segmentation, the output is
a region or a structure that collectively covers the entire image. These regions have
similar characteristics including colors, texture, or intensity. Image segmentation is
important because it can enhance the analysis of images with more granularity. It
extracts the objects of interest, for further processing such as description or
recognition.
Image segmentation works by using encoders and decoders. Encoders take in
input, which is a raw image, it then extracts features, and finally, decoders generate
an output which is an image with segments.

## Dataset:
Cityscapes data contains labeled videos taken from vehicles driven in Germany.
The dataset has images from the original videos, and the semantic segmentation
labels are shown in images alongside the original image.
This dataset has 1500 training image files and 200 test image files. Each image file
is 256x512 pixels, and each file is a composite with the original photo on the left
half of the image, alongside the labeled (mask) image (output of semantic
segmentation) on the right half.

I use U-Net architecture with
convolutional layers to build a model for the segmentation task.

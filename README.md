# Research Project
The below codes are implemented by Pytorch for personal study.

## Face Super-Resolution
Deokyun Kim, Minseon Kim, Gihyun Kwon and Dae-Shik Kim, [Progressive Face Super-Resolution via Attention to Facial Landmark](https://arxiv.org/abs/1908.08239), BMVC 2019

**Abstract:** *Face Super-Resolution (SR) is a subfield of the SR domain that specifically targets the reconstruction of face images. The main challenge of face SR is to restore essential facial features without distortion. We propose a novel face SR method that generates photo-realistic 8× super-resolved face images with fully retained facial details. To that end, we adopt a progressive training method, which allows stable training by splitting the network into successive steps, each producing output with a progressively higher resolution. We also propose a novel facial attention loss and apply it at each step to focus on restoring facial attributes in greater details by multiplying the pixel difference and heatmap values. Lastly, we propose a compressed version of the state-of-the-art face alignment network (FAN) for landmark heatmap extraction. With the proposed FAN, we can extract the heatmaps suitable for face SR and also reduce the overall training time. Experimental results verify that our method outperforms state-of-the-art methods in both qualitative and quantitative measurements, especially in perceptual quality.*


Github(Official): https://github.com/DeokyunKim/Progressive-Face-Super-Resolution



## Single Image Super-Resolution (SISR)

To learn for Single image super-resolution using deep learning, I read the following typical papers related to SISR below and implemented in code.

Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158), CVPR 2016.

Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802), CVPR 2017 Oral presentation.

## Translation from Near Infra-Red Image to RGB Image

I am trying to solve image translation from NIR Images to RGB Images.
The key question is

1. How can deep neural network infer the color information from NIR Images?

2. Following question is "shouldn't the same object be inferred from different colors, depending conditions such as weather and time?"

In addition, research so far has found that the deep neural network is biased into green(wood) and blue(sky) that make up the majority of the images.

Patricia L. Suarez, Angel D. Sappa, and Boris X. Vintimilla, [Infrared Image Colorization based on a Triplet DCGAN Architecture](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w3/papers/Suarez_Infrared_Image_Colorization_CVPR_2017_paper.pdf), CVPR 2017 Workshop


## One-Shot Image Quality Enhancement

Visual correction of photographs is the domain of well-trained experts. I trained a deep neural network with noisy and well-corrected clean image by referring to the following papers. Because the noisy image is mostly dark, and clean image is mostly bright, I have found that the trained deep neural network makes the bright images much brighter. Thus, I trained the network by using the color-wise normalize method to robust the brightness of images.

Andrey Ignatov , Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey, Luc Van Gool, [DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks](http://www.vision.ee.ethz.ch/~timofter/publications/Ignatov-ICCV-2017.pdf), ICCV 2017

Yu-Sheng Chen, Yu-Ching Wang,Man-Hsin Kao, Yung-Yu Chuang, [Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs](https://www.cmlab.csie.ntu.edu.tw/project/Deep-Photo-Enhancer/CVPR-2018-DPE.pdf), CVPR 2018

MIT-Adobe 5K Dataset [Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs](https://data.csail.mit.edu/graphics/fivek/)

## Object Segmentation



Jonathan Long, Evan Shelhamer, Trevor Darrell, [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), CVPR 2015

[Visual Object Classes Challenge 2012 (PASCAL VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

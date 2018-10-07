# DogClassifier

A model trained to classifiy dog breeds. This project takes data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification). This repository contains two separate models one which attempts to impelent an inception style network and a second that makes use of the pretrained [Inception ResNet v2](https://ai.googleblog.com/2016/08/improving-inception-and-image.html) model that is available in [Keras](https://keras.io/applications/#inceptionresnetv2).

The training and test images were cropped to better focus on the dogs in the images. This process was completed using a modified version of the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to automate this process.

## Navigation

* [Homemade inception model](./homemade_Inception/README.md)

* [Pretrained model](./pretrained/README.md)


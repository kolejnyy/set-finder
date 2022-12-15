# Find a Set

A tool for finding a set of cards in a picture of a table.

## TODO
- [ ] Develop a neural network for card detection
- [ ] Optimize speed, so that the tool can be used in real time 

## Overview
The model consists of two independent parts: detection and recognition engines. The detection engine is responsible for finding the cards on the table, while the recognition engine is responsible for recognizing the cards. The detection engine uses more classical image processing approach utilizing the OpenCV library, while the recognition engine is based on a Convolutional Neural Network architecture.

## Usage
```
from models.recognizer import Recognizer
from models.detector import Detector
from models.setfinder import SetFinder

# Initialize the models
detector = Detector(180, save=True)
recognizer = Recognizer(256)
recognizer.load_state_dict(torch.load('weights/finetuned.pth'))
setfinder = SetFinder(detector, recognizer, 256, True)

# Specify the path to the image
path = 'dataset/full_photo/IMG_2779.JPG'
sets = setfinder.find_sets(path)
```

## Detection engine

The detection engine is based on the OpenCV library. The algorithm is as follows:
- The input image is converted to grayscale and binarized using a

## Recognition engine

Results of traninig different models on 1000 dynamically generated mini-batches consisting of 32 samples (single-card images with distinct backgrounds of size 512x512) each:
* 5 convolutional layers (3/16/32/64/64): ~2.6 loss
* 6 convolutional layers (3/16/32/64/128/128): ~2.3 loss
* 7 convolutional layers (3/16/32/64/128/256/256): 
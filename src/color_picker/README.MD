# The colour picker system 
This repository contains three scripts that make up the system for the colour picker stimuli.

# scripts
- color_selector_gui.py
    This is a stand-alone module that helps to find the colour of specific pixels in a given image with a GUI, given that this is sometimes difficult with code alone when we have a specific image to test.

- generate_dataset.py
    This is a train image generation script where the two input parameters are the save path of the images and the number of images to generate. Each stimulus is composed of random shapes and an arrow at a random location. Each image is labelled with the grayscale colour of the pixel under the arrow.

- color_picker_test_stimuli.py
    This script takes:
    1. an image
    2. a colour of interest
    This script then makes {the number of pixels of the colour of interest} on the image a number of images, with each image having an arrow on a distinct pixel; these images can then be subjected to any trained models (on the train stimuli).
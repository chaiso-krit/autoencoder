#!/usr/bin/python
#
# Usage: ./create_mat_from_folder.py IMAGE_FOLDER OUTPUT_FILE
# 
# Author : Krit Chaiso
#

import cv2
import os
import numpy
import sys

import scipy.io as sio

image_extensions = ['.pgm', '.png', '.jpg', '.jpeg']

def get_image_list(folder_name = 'input'):
    # get all images from image_folder
    images_filename = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith(tuple(image_extensions)):
                images_filename.append(os.path.join(root, file))
    return images_filename

def normalize(array, is_sigmoid = False):
    mean = numpy.mean(array)
    array = array - mean
    pstd = 3 * numpy.std(array)
    array = numpy.array([max(min(pixel, pstd), -pstd)/pstd for pixel in array])
    if is_sigmoid:
        array = (array+1)*0.4+0.1
    else:
        array = array*0.9
    return array

if __name__ == "__main__":
    if len(sys.argv) == 3:
        IMAGE_FOLDER = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
    else:
        print "Usage: ./create_mat_from_folder.py IMAGE_FOLDER OUTPUT_FILE"
        exit(0)

    image_list = get_image_list(IMAGE_FOLDER)

    train = []
    # create dataset
    for image in image_list:
        im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        h,w = im.shape
        im_array = im.reshape(h*w)
        im_array = normalize(im_array, is_sigmoid=True)
        train.append(im_array)

    print len(train)
    sio.savemat(OUTPUT_FILE, {'train_data':numpy.vstack(train)})


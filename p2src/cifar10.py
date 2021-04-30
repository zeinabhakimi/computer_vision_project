import os
import re
import sys
import argparse
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # matplotlib inline
import pickle
from IPython.display import Image, display

import PIL.Image
from io import StringIO


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_images(path, wildcard):
    images_data = []

    # for f in fnmatch.filter(os.listdir(FLAGS.image_set), 'data_batch_*'):
    #     images_raw = unpickle(FLAGS.image_set + f)
    #     if len(images_data) == 0:
    #         images_data = images_raw[b'data']
    #         images_label = np.asarray(images_raw[b'labels'], dtype='uint8')
    #     else:
    #         images_data = np.concatenate((images_data, images_raw[b'data']))
    #         images_label = np.append(images_label, np.asarray(
    #             images_raw[b'labels'], dtype='uint8'))

    images_raw = unpickle(path + 'test_batch')
    images_data = images_raw[b'data']
    images_label = np.asarray(images_raw[b'labels'], dtype='uint8')

    images = np.transpose(np.reshape(images_data, (-1, 32, 32, 3),
                                     order='F'), axes=(0, 2, 1, 3))  # order batch,x,y,color
    return images, images_label

# def load_images():
#     images_data = []
#
#     for f in fnmatch.filter(os.listdir(FLAGS.image_set), 'data_batch_*'):
#         images_raw = unpickle(FLAGS.image_set + f)
#         if len(images_data) == 0:
#             images_data = images_raw[b'data']
#             images_label = np.asarray(images_raw[b'labels'], dtype='uint8')
#         else:
#             images_data = np.concatenate((images_data, images_raw[b'data']))
#             images_label = np.append(images_label, np.asarray(
#                 images_raw[b'labels'], dtype='uint8'))
#
#     test_raw = unpickle(FLAGS.image_set + 'test_batch')
#     test_data = test_raw[b'data']
#     test_label = np.asarray(test_raw[b'labels'], dtype='uint8')
#
#     images_data = np.transpose(np.reshape(images_data, (-1, 32, 32, 3),
#                                           order='F'), axes=(0, 2, 1, 3))
#
#     test = np.transpose(np.reshape(test_data, (-1, 32, 32, 3),
#                                    order='F'), axes=(0, 2, 1, 3))
#
#     return images, images_label, test, test_label

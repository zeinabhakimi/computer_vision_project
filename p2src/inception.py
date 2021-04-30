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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile


class Inception:

    # Name of the tensor for feeding the decoded input image.
    tensor_name_input_image = "DecodeJpeg:0"

    # Name of the tensor for the resized input image.
    tensor_name_resized_image = "ResizeBilinear:0"

    # Name of the tensor for the output of the softmax-classifier.
    tensor_name_softmax = "softmax:0"

    # Name of the tensor for the output of the Inception model.
    tensor_name_transfer_layer = "pool_3:0"

    def __init__(self, model_dir, graph_filename):

        self.graph = tf.Graph()

        with self.graph.as_default():
            # Creates a graph from saved GraphDef file and returns a saver
            graph_path = os.path.join(model_dir, graph_filename)
            with tf.gfile.FastGFile(graph_path, 'rb') as f:

                graph_def = tf.GraphDef()

                graph_def.ParseFromString(f.read())

                _ = tf.import_graph_def(graph_def, name='')

        self.resized_image = self.graph.get_tensor_by_name(
            self.tensor_name_resized_image)

        self.transfer_layer = self.graph.get_tensor_by_name(
            self.tensor_name_transfer_layer)

        self.session = tf.Session(graph=self.graph)

    def close(self):
        self.session.close()

    def _write_summary(self, logdir='tflog/'):
        writer = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
        writer.close()

    def _create_feeddict(self, image=None):

        if image is not None:
            feed_dict = {self.tensor_name_input_image: image}
        else:
            raise ValueError("image must be set.")

        return feed_dict

    def get_resized_image(self, image=None):

        feed_dict = self._create_feeddict(image=image)

        resized_image = self.session.run(self.resized_image, feed_dict=feed_dict)

        resized_image = resized_image.squeeze(axis=0)

        resized_image = resized_image.astype('uint8')

        return resized_image

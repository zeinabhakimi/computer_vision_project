#!/usr/bin/env python3
__author__ = 'Morteza Ramezani'

import os
import re
import sys
import argparse
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from IPython.display import Image, display


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile

# import nodelookup

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_images():
    images_data = []

    for f in fnmatch.filter(os.listdir(FLAGS.image_set), 'data_batch_*'):
        images_raw = unpickle(FLAGS.image_set + f)
        if len(images_data) == 0:
            images_data = images_raw[b'data']
            images_label = np.asarray(images_raw[b'labels'], dtype='uint8')
        else:
            images_data = np.concatenate((images_data, images_raw[b'data']))
            images_label = np.append(images_label, np.asarray(
                images_raw[b'labels'], dtype='uint8'))
        break

    # images_raw = unpickle(FLAGS.image_set + 'test_batch')
    # images_data = images_raw[b'data']
    # images_label = np.asarray(images_raw[b'labels'], dtype='uint8')

    images = np.transpose(np.reshape(images_data, (-1, 32, 32, 3),
                                     order='F'), axes=(0, 2, 1, 3))  # order batch,x,y,color
    return images, images_label


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    summary_writer = tf.summary.FileWriter('./tflog', graph=tf.Session().graph)


def extract_features(images, images_label):
    nb_features = 2048
    nb_softmax = 1008
    features = np.empty((len(images), nb_features))
    preds = np.empty((len(images), nb_softmax))
    labels = []

    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, img_data in enumerate(images):

            if (ind == 100):
                break

            if (ind % 100 == 0):
                print('Processing %s...' % (ind))


            representation = sess.run(representation_tensor, {
                                      'DecodeJpeg:0': img_data})
            features[ind, :] = np.squeeze(representation)


            # [representation, prediction] = sess.run([representation_tensor, softmax_tensor], {
            #     'DecodeJpeg:0': img_data})
            # preds[ind, :] = np.squeeze(prediction)
            # labels.append(images_label[ind])
            # labels.append(re.split('_\d+', image.split('/')[1])[0])
            # node_lookup = NodeLookup()
            # temp_predict = np.squeeze(prediction)
            # top_k = temp_predict.argsort()[-5:][::-1]
            # for node_id in top_k:
            #     human_string = node_lookup.id_to_string(node_id)
            #     score = temp_predict[node_id]
            #     print('%s (score = %.5f)' % (human_string, score))
            # print('--'*50)

    return features, images_label


def main(_):
    images, images_label = load_images()
    create_graph()
    features, labels = extract_features(images, images_label)
    pickle.dump(features, open('features_15', 'wb'))
    pickle.dump(images_label, open('labels_15', 'wb'))


if __name__ == '__main__':
    labels_name = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_set',
        type=str,
        default='./cifar-10-batches-py/',
        help='Absolute path to image file.'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='./inception/',
        help="""\
     Path to classify_image_graph_def.pb,
     imagenet_synset_to_human_label_map.txt, and
     imagenet_2012_challenge_label_map_proto.pbtxt.\
     """
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

#!/usr/bin/env python3
__author__ = 'Morteza Ramezani'

import os
import re
import sys
import argparse
import fnmatch
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from IPython.display import Image, display


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split

from cifar10 import *


def load_pool3_data():
    X_data_pool3 = pickle.load(open(FLAGS.features + '/features55', 'rb'))
    y_data_pool3 = pickle.load(open(FLAGS.features + '/labels55', 'rb'))
    X_test_pool3 = pickle.load(open(FLAGS.features + '/features_test', 'rb'))
    y_test_pool3 = pickle.load(open(FLAGS.features + '/labels_test', 'rb'))

    return X_data_pool3, y_data_pool3, X_test_pool3, y_test_pool3

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_2layers(graph, class_count, final_tensor_name, bottleneck_tensor):

    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, 2048],
        name='BottleneckInputPlaceholder')

    layer_weights = tf.Variable(
        tf.truncated_normal([2048, class_count], stddev=0.001), name='final_weights')
    variable_summaries(layer_weights)

    layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    variable_summaries(layer_weights)

    logits = tf.matmul(bottleneck_input, layer_weights,
                       name='final_matmul') + layer_biases
    tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    ground_truth_placeholder = tf.placeholder(
        tf.float32, [None, class_count], name='GroundTruthInput')

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=ground_truth_placeholder)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy_mean)

    return train_step, cross_entropy_mean, final_tensor, bottleneck_input, ground_truth_placeholder


def add_evaluation_step(graph, final_tensor_name, ground_truth_tensor):

    correct_prediction = tf.equal(
        tf.argmax(final_tensor_name, 1), tf.argmax(ground_truth_tensor, 1))

    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    tf.summary.scalar('accuracy', evaluation_step)

    return evaluation_step


def iterate_mini_batches(X_input, Y_input, batch_size):
    n_train = X_input.shape[0]
    for ndx in range(0, n_train, batch_size):
        yield X_input[ndx:min(ndx + batch_size, n_train)], Y_input[ndx:min(ndx + batch_size, n_train)]

def do_train(sess, X_train_pool3, y_train_pool3, X_validation_pool3, y_validation_pool3, X_test_pool3, y_test_pool3):

    mini_batch_size = 10

    n_train = X_train_pool3.shape[0]

    graph, bottleneck_tensor = create_graph()

    train_step, cross_entropy, final_tensor, bottleneck_input, ground_truth_tensor = add_final_2layers(
        graph, len(labels_name), FLAGS.final_tensor_name,
        bottleneck_tensor)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    init = tf.global_variables_initializer()
    sess.run(init)

    evaluation_step = add_evaluation_step(
        graph, final_tensor, ground_truth_tensor)


    summary_writer = tf.summary.FileWriter('./tflog', graph=tf.Session().graph)

    training_accuracy_hist = np.array([])

    i = 0
    epocs = 1
    for epoch in range(epocs):
        shuffledRange = np.random.permutation(n_train)

        y_one_hot_train = np.eye(len(labels_name))[y_train_pool3]
        y_one_hot_validation = np.eye(len(labels_name))[y_validation_pool3]

        shuffledX = X_train_pool3[shuffledRange, :]
        shuffledY = y_one_hot_train[shuffledRange]

        for Xi, Yi in iterate_mini_batches(shuffledX, shuffledY, mini_batch_size):

            train_summary, _ = sess.run([merged, train_step],
                     feed_dict={bottleneck_input: Xi,
                                ground_truth_tensor: Yi})
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == FLAGS.num_training_steps)

            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:

                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: Xi,
                               ground_truth_tensor: Yi})

                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: X_validation_pool3,
                               ground_truth_tensor: y_one_hot_validation})
                validation_writer.add_summary(validation_summary, i)

                print('%s: Step %d: Train accuracy = %.1f%%, Cross entropy = %f, Validation accuracy = %.1f%%' %
                      (datetime.now(), i, train_accuracy * 100, cross_entropy_value, validation_accuracy * 100))

                if len(training_accuracy_hist) == 0:
                    training_accuracy_hist = np.asarray([[train_accuracy], [cross_entropy_value], [validation_accuracy]])
                else:
                    training_accuracy_hist = np.append(training_accuracy_hist, [[train_accuracy], [cross_entropy_value], [validation_accuracy]], axis=1)


            i += 1

    test_accuracy = sess.run(
        evaluation_step,
        feed_dict={bottleneck_input: X_test_pool3,
                   ground_truth_tensor: np.eye(len(labels_name))[y_test_pool3]})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

    np.save('./results/training_accuracy', training_accuracy_hist)
    np.save('./results/test_accuracy', test_accuracy)


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    with tf.Session() as sess:
        with tf.gfile.FastGFile(os.path.join(
                FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = tf.import_graph_def(graph_def, name='', return_elements=[
                'pool_3/_reshape:0', 'DecodeJpeg/contents:0', 'ResizeBilinear:0'])
    # summary_writer = tf.summary.FileWriter('./tflog', graph=tf.Session().graph)
    return sess.graph, bottleneck_tensor


def main(_):
    sess = tf.InteractiveSession()
    X_data_pool3, y_data_pool3, X_test_pool3, y_test_pool3 = load_pool3_data()
    X_train_pool3, X_validation_pool3, y_train_pool3, y_validation_pool3 = train_test_split(
        X_data_pool3, y_data_pool3, test_size=0.20, random_state=42)

    do_train(sess, X_train_pool3, y_train_pool3, X_validation_pool3,
             y_validation_pool3, X_test_pool3, y_test_pool3)


if __name__ == '__main__':
    labels_name = np.array(['airplane', 'automobile', 'bird', 'cat',
                            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_set',
        type=str,
        default='./cifar-10-batches-py',
        help='Absolute path to image file.'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='./inception',
        help="""\
     Path to classify_image_graph_def.pb,
     imagenet_synset_to_human_label_map.txt, and
     imagenet_2012_challenge_label_map_proto.pbtxt.\
     """
    )

    parser.add_argument(
        '--features',
        type=str,
        default='./results',
        help=""""""
    )

    parser.add_argument(
        '--num_training_steps',
        type=int,
        default=100,
        help="""How many training steps to run before ending."""
    )

    parser.add_argument(
        '--learning_rate',
        type=int,
        default=0.01,
        help="""How large a learning rate to use when training."""
    )

    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""The name of the output classification layer in he retrained graph."""
    )

    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help="""How often to evaluate the training results."""
    )

    parser.add_argument(
      '--summaries_dir',
      type=str,
      default='./tflog',
      help='Where to save summary logs for TensorBoard.'
  )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

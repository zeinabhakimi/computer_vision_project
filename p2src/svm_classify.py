#!/usr/bin/env python3
__author__ = 'Morteza Ramezani'

import os
import re
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # matplotlib inline
import pickle

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC


def plot_confusion_matrix(y_true, y_pred):
    cm_array = confusion_matrix(y_true, y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()


def main():
    # features = pickle.load(open('features55', 'rb'))
    # labels = pickle.load(open('labels', 'rb'))
    # X_train, X_test, y_train, y_test = train_test_split(
        # features, labels, test_size=0.2, random_state=42)

    X_train = pickle.load(open('./results/features55', 'rb'))
    y_train = pickle.load(open('./results/labels55', 'rb'))

    X_test = pickle.load(open('./results/features_test', 'rb'))
    y_test = pickle.load(open('./results/labels_test', 'rb'))

    clf = LinearSVC(C=1.0, loss='squared_hinge',
                    penalty='l2', multi_class='ovr')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # count = 0
    # for idx, y in enumerate(y_pred):
    #     if y_test[idx] == y:
    #         count += 1
    # print(count, len(y_pred), count / len(y_pred) * 1.0)

    print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
    np.save('./results/svm_test_res', y_test)
    np.save('./results/svm_pred_res', y_pred)

    # plot_confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    main()

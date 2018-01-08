"""Utility functions for loading and plotting data."""

import csv
import numpy as np
import matplotlib.pyplot as plt


def load_reviews_data(reviews_data_path):
    """Loads the reviews dataset as a list of dictionaries.

    Arguments:
        reviews_data_path(str): Path to the reviews dataset .csv file.

    Returns:
        A list of dictionaries where each dictionary maps column name
        to value for a row in the reviews dataset.
    """
    data = []
    with open(reviews_data_path, encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data


def load_toy_data(toy_data_path):
    """Loads the 2D toy dataset as numpy arrays.

    Arguments:
        toy_data_path(str): Path to the toy dataset .csv file.

    Returns:
        A tuple (features, labels) in which features is an Nx2 numpy
        matrix and labels is a length-N vector of +1/-1 labels.
    """
    data = np.loadtxt(toy_data_path)
    X = data[:,1:]
    y = data[:,0]
    return X, y


def plot_toy_data(data, labels):
    """Plots the toy data in 2D.

    Arguments:
        data(ndarray): An Nx2 ndarray of points.
        labels(ndarray): A length-N vector of +1/-1 labels.
    """
    data1 = data[labels==-1]
    data2 = data[labels==1]
    plt.scatter(data1[:, 0], data1[:, 1], color='red')
    plt.scatter(data2[:, 0], data2[:, 1], color='blue')
    x = [3.2, -2]
    y = [-2, 4]
    plt.plot(x, y)
    plt.show()
    return

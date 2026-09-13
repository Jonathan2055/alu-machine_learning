#!/usr/bin/env python3
"""Creates placeholders for a neural network."""


import tensorflow as tf
"""Creates placeholders for a neural network."""


def create_placeholders(nx, classes):
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
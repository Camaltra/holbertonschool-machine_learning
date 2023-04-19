#!/usr/bin/env python3

"""useless comments"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluate a saved model
    :param X: The data set
    :param Y: The thruth label
    :param save_path: The path of the saved model
    :return: Network’s prediction, accuracy, and loss, respectivel
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(session, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        prediction = session.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
        loss = session.run(loss, feed_dict={x: X, y: Y})

    return prediction, accuracy, loss

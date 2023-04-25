#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf

suffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        batch_size=32,
        epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt"
):
    """
    :param X_train: Is a numpy.ndarray of shape
                    (m, 784) containing the training data
    :param Y_train: Is a one-hot numpy.ndarray
                    of shape (m, 10) containing the training labels
    :param X_valid: Is a numpy.ndarray of shape
                    (m, 784) containing the validation data
    :param Y_valid: Is a one-hot numpy.ndarray
                    of shape (m, 10) containing the validation labels
    :param batch_size: Is the number of data points in a batch
    :param epochs: Is the number of times the training
                   should pass through the whole dataset
    :param load_path: Is the path from which to load the model
    :param save_path: Is the path to where the
                      model should be saved after training
    :return: The path where the model was saved
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(session, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]

        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch_idx in range(epochs + 1):
            cost_train = session.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = session.run(accuracy,
                                         feed_dict={x: X_train, y: Y_train})
            cost_val = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = session.run(accuracy,
                                       feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch_idx))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            if epoch_idx == epochs:
                return saver.save(session, save_path)

            x_train, y_train = suffle_data(X_train, Y_train)
            dataset_len = x_train.shape[0]
            for i in range(0, dataset_len + 1, batch_size):
                start = i
                end = i + batch_size if i + batch_size <= dataset_len else -1
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]
                feed_dict = {x: x_batch, y: y_batch}
                session.run(train_op, feed_dict=feed_dict)
                if (i / batch_size) % 100 == 0 and i != 0:
                    loss_mini_batch = session.run(loss, feed_dict=feed_dict)
                    accuracy_mini_batch = session.run(accuracy,
                                                      feed_dict=feed_dict)
                    print("\tStep {}:".format((i / batch_size)))
                    print("\t\tCost: {}".format(loss_mini_batch))
                    print("\t\tAccuracy: {}".format(accuracy_mini_batch))

#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf

suffle_data = __import__('2-shuffle_data').shuffle_data


def verbose_mini_batch(session, step, loss, accuracy, feed_dict):
    loss_mini_batch = session.run(loss, feed_dict=feed_dict)
    accuracy_mini_batch = session.run(accuracy,
                                      feed_dict=feed_dict)
    print("\tStep {}:".format(step))
    print("\t\tCost: {}".format(loss_mini_batch))
    print("\t\tAccuracy: {}".format(accuracy_mini_batch))


def verbose_epoch(
        session,
        epoch_idx,
        x,
        y,
        loss,
        accuracy,
        x_train,
        y_train,
        x_test,
        y_test
):
    cost_train = session.run(loss, feed_dict={x: x_train, y: y_train})
    accuracy_train = session.run(accuracy,
                                 feed_dict={x: x_train, y: y_train})
    cost_val = session.run(loss, feed_dict={x: x_test, y: y_test})
    accuracy_val = session.run(accuracy,
                               feed_dict={x: x_test, y: y_test})
    print("After {} epochs:".format(epoch_idx))
    print("\tTraining Cost: {}".format(cost_train))
    print("\tTraining Accuracy: {}".format(accuracy_train))
    print("\tValidation Cost: {}".format(cost_val))
    print("\tValidation Accuracy: {}".format(accuracy_val))


def train_mini_batch(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=32,
        epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt"
):
    """
    :param x_train: Is a numpy.ndarray of shape
                    (m, 784) containing the training data
    :param y_train: Is a one-hot numpy.ndarray
                    of shape (m, 10) containing the training labels
    :param x_test: Is a numpy.ndarray of shape
                    (m, 784) containing the validation data
    :param y_test: Is a one-hot numpy.ndarray
                    of shape (m, 10) containing the validation labels
    :param batch_size: Is the number of data points in a batch
    :param epochs: Is the number of times the training
                   should pass through the whole dataset
    :param load_path: Is the path from which to load the model
    :param save_path: Is the path to where the
                      model should be saved after training
    :return: The path where the model was saved
    """
    with tf.compat.v1.Session() as session:
        saver = tf.compat.v1.train.import_meta_graph(load_path + ".meta")
        saver.restore(session, load_path)

        x = tf.compat.v1.get_collection('x')[0]
        y = tf.compat.v1.get_collection('y')[0]

        accuracy = tf.compat.v1.get_collection('accuracy')[0]
        loss = tf.compat.v1.get_collection('loss')[0]
        train_op = tf.compat.v1.get_collection('train_op')[0]

        for epoch_idx in range(epochs):
            verbose_epoch(session, epoch_idx, x, y, loss,
                          accuracy, x_train, y_train, x_test, y_test)

            x_train, y_train = suffle_data(x_train, y_train)
            dataset_len = x_train.shape[0]
            steps = [(i, i + batch_size) for i in
                     range(0, dataset_len, batch_size)]
            for step, (start, end) in enumerate(steps, start=1):
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]
                feed_dict = {x: x_batch, y: y_batch}
                session.run(train_op, feed_dict=feed_dict)
                if step % 100 == 0:
                    verbose_mini_batch(session, step,
                                       loss, accuracy, feed_dict)

        verbose_epoch(session, epoch_idx, x, y, loss,
                      accuracy, x_train, y_train, x_test, y_test)

        return saver.save(session, save_path)

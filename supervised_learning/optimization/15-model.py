#!/usr/bin/env python3

"""useless comment"""

import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
    Create a layer that normalized the unactivate input data
    :param prev: The prev layers ouput
    :param n: The number of node in the layer
    :param activation: The activation fuction
    :return: The new created layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense_layer = tf.layers.Dense(units=n,
                                  kernel_initializer=init)

    output = dense_layer(prev)

    mean, variance = tf.nn.moments(output, axes=[0])

    scale = tf.Variable(tf.ones([n]))
    shift = tf.Variable(tf.zeros([n]))

    epsilon = 1e-8
    output = tf.nn.batch_normalization(output, mean, variance,
                                       shift, scale, epsilon)

    if activation is not None:
        output = activation(output)

    return output


def create_layer(prev, n, activation):
    """
    Create placeholders tensor
    :param nx: The number of feature columns in our data
    :param classes: The number of classes in our classifier
    :return: The two placeholders
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer(prev)


def create_placeholders(nx, classes):
    """
    Create placeholders tensor
    :param nx: The number of feature columns in our data
    :param classes: The number of classes in our classifier
    :return: The two placeholders
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y


def forward_prop(x, layers=[], activations=[]):
    """
    Compute the forward propagation of the NN
    :param x: The input layer
    :param layer_sizes: A list contain number of node for
                        each layers
    :param activations: A list contain the activation
                        function for each layer
    :return: The final tensor
    """
    model_length = len(layers)
    for idx, (layer, activation) in enumerate(zip(layers, activations)):
        if idx == 0:
            predictions = create_layer(x, layer, activation)
        if idx == len(model_length) - 1:
            predictions = create_layer(predictions, layer, activation)
        else:
            predictions = create_batch_norm_layer(predictions, layer,
                                                  activation)
    return predictions


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Define the training operator using Adam optimizer
    :param loss: The loss function
    :param alpha: The learning rate
    :param beta1: The momentum weight
    :param beta2: The RMS weigth
    :param epsilon: A small number to not divide by 0
    :return: The train op
    """
    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Calculate the new learning rate depend on a decay value and a step
    as the alpha should be updated in a step-wise fashion
    :param alpha: The initial learning rate
    :param decay_rate: The decay rate
    :param global_step: The actual step
    :param decay_step: The step-wise indicator
    :return: The new learning rate
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def shuffle_data(X, Y):
    """
    Shuffle a dataset (X and Y set in the same ways)
    :param X: The dataset feature
    :param Y: The labels
    :return: The shuffled datasets
    """
    dataset_len = X.shape[0]
    indices_permutted = np.random.permutation(dataset_len)
    return X[indices_permutted], Y[indices_permutted]


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
    """
    Verbose the epoch -- for all arg see in the main function
    (Typing would be better here)
    """
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


def verbose_mini_batch(session, step, loss, accuracy, feed_dict):
    """
    Verbose the minibatch -- for all arg see in the main function
    (Typing would be better here)
    """
    loss_mini_batch = session.run(loss, feed_dict=feed_dict)
    accuracy_mini_batch = session.run(accuracy,
                                      feed_dict=feed_dict)
    print("\tStep {}:".format(step))
    print("\t\tCost: {}".format(loss_mini_batch))
    print("\t\tAccuracy: {}".format(accuracy_mini_batch))


def calculate_accuracy(y, y_preds):
    """
    As the function say, calculate the accuracy
    :param y: The thuth label
    :param y_pred: The predicted label
    :return: The tensof of predicted value
    """
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_preds, 1))
    return tf.reduce_mean(tf.cast(accuracy, tf.float32))


def calculate_loss(y, y_pred):
    """
    As the function say, calculate the loss
    :param y: The thuth label
    :param y_pred: The predicted label
    :return: The tensof of predicted value
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    Build, train and save a neural network model in tensorflow using
    :param Data_train: A tuple containing the training data
    :param Data_valid: A tuple containing the validation data
    :param layers: A list containing the number of nodes in each layer
    :param activations: A list containing the activation function
    :param alpha: The learning rate
    :param beta1: The momentum weight
    :param beta2: The RMSProp weight
    :param epsilon: A small number to avoid division by zero
    :param decay_rate: The decay rate for inverse time decay
    :param batch_size: The size of mini batch
    :param epochs: The number of epochs
    :param save_path: The path to save the model
    :return: The path where the model was saved
    """
    x_train, y_train = Data_train
    x_valid, y_valid = Data_valid

    x, y = create_placeholders(x_train.shape[1],
                               y_train.shape[1])
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha_decay = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha_decay, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)

        for epoch_idx in range(epochs + 1):
            verbose_epoch(session, epoch_idx, x, y, loss,
                          accuracy, x_train, y_train, x_valid, y_valid)

            if epoch_idx == epochs:
                return saver.save(session, save_path)

            session.run(global_step.assign(epoch_idx))
            session.run(alpha_decay)

            x_t, y_t = shuffle_data(x_train, y_train)
            dataset_len = x_t.shape[0]
            steps = [(i, i + batch_size) for i in
                     range(0, dataset_len, batch_size)]
            for step, (start, end) in enumerate(steps, start=1):
                x_batch = x_t[start:end]
                y_batch = y_t[start:end]
                feed_dict = {x: x_batch, y: y_batch}
                session.run(train_op, feed_dict=feed_dict)
                if step % 100 == 0:
                    verbose_mini_batch(session, step,
                                       loss, accuracy, feed_dict)

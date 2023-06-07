#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf
import os
import numpy as np


def sigmoid(x):
    """
    Sigmoid function
    :param x: The x parameter
    :return: The computed function given x
    """
    return 1 / (1 + np.exp(-x))


def get_classes_name(classes_name_path):
    """
    Get the classes name form a given .txt. file
    :param classes_name_path: The path to the .txt file
    :return: And array that contain class name
    """
    classes_name = []
    with open(classes_name_path, "r") as f:
        for line in f.readlines():
            classes_name.append(line.strip())

    return classes_name


class Yolo:
    """
    Yolo model class
    """
    def __init__(
            self,
            model_path,
            classes_name_path,
            class_t,
            nms_t,
            anchors
    ):
        """
        Init the class
        :param model_path: The path to the DarkNet model
        :param classes_name_path: The path to the file cotaining classes names
        :param class_t: The box threshold for the initial filtrering step
        :param nms_t: The IOU threshold for non-max suppression
        :param anchors: The anchor boxes
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("Wrong model file path")

        if not os.path.exists(classes_name_path):
            raise FileNotFoundError("Wrong classes file path")

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = get_classes_name(classes_name_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs of the Darknet YOLO model to get all the
        boundary boxes for each output, and each cell of output, and
        each anchor boxes
        :param outputs: The output of the YOLO model
        :param image_size: The original image size
        :return: A processed output
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output_idx, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            c_x = np.arange(grid_width)
            c_x = np.tile(c_x, grid_height)
            c_x = c_x.reshape(grid_height, grid_width, 1)

            c_y = np.arange(grid_height)
            c_y = np.tile(c_y, grid_width)
            c_y = c_y.reshape(1, grid_height, grid_width).T

            bx = (sigmoid(t_x) + c_x) / grid_width
            by = (sigmoid(t_y) + c_y) / grid_height
            bw = np.exp(t_w) * self.anchors[output_idx, :, 0]
            bw /= self.model.input.shape[1].value
            bh = np.exp(t_h) * self.anchors[output_idx, :, 1]
            bh /= self.model.input.shape[2].value

            y1 = (by - bh / 2) * image_height
            x1 = (bx - bw / 2) * image_width
            x2 = (bw / 2 + bx) * image_width
            y2 = (bh / 2 + by) * image_height

            b_size = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            b_size[:, :, :, 0] = x1
            b_size[:, :, :, 1] = y1
            b_size[:, :, :, 2] = x2
            b_size[:, :, :, 3] = y2
            boxes.append(b_size)

            box_confidences.append(sigmoid(output[:, :, :, 4:5]))

            box_class_probs.append(sigmoid(output[:, :, :, 5:]))

        return boxes, box_confidences, box_class_probs

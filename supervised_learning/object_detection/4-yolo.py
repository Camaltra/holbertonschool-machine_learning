#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf
import os
import numpy as np
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes that are under a certain proba threshold
        :param boxes: The boxes (center, witdh height)
        :param box_confidences: The confidences of a box
        :param box_class_probs: The proba for each classes
        :return: filtered_boxes: a numpy.ndarray of shape (?, 4)
                                 containing all of the filtered bounding boxes:
                 box_classes: a numpy.ndarray of shape
                              (?,) containing the class
                              number that each box in filtered_boxes
                              predicts, respectively
                 box_scores: a numpy.ndarray of shape (?) containing
                             the box scores for each box in
                             filtered_boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for boxes_i, box_preds in enumerate(box_confidences):
            height, width, anchors, _ = box_preds.shape
            for h_i in range(height):
                for w_i in range(width):
                    for anchor in range(anchors):
                        current_condifance = box_preds[h_i, w_i, anchor, 0]
                        current_boxes = boxes[boxes_i][h_i, w_i, anchor]
                        classe = np.argmax(
                            box_class_probs[boxes_i][h_i, w_i, anchor]
                        )
                        classe_proba = np.max(
                            box_class_probs[boxes_i][h_i, w_i, anchor]
                        )
                        box_score = current_condifance * classe_proba
                        if box_score >= self.class_t:
                            filtered_boxes.append(current_boxes)
                            box_classes.append(classe)
                            box_scores.append(box_score)

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores

    def _nms(self, filtered_class_boxes, filtered_class_box_score):
        """
        Sub function to compute the non-max supression for a given class
        :param filtered_class_boxes: The filtered classes boxes of a given
                                     class
        :param filtered_class_box_score: The score to sort on it
        :return: The index of the non overlaping box
        """
        keeped_boxes = []

        x1 = filtered_class_boxes[:, 0]
        y1 = filtered_class_boxes[:, 1]
        x2 = filtered_class_boxes[:, 2]
        y2 = filtered_class_boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = filtered_class_box_score.argsort()

        while idxs.size != 0:
            last = idxs.size - 1
            current_boxes_idx = idxs[last]
            keeped_boxes.append(current_boxes_idx)

            xx1 = np.maximum(x1[current_boxes_idx], x1[idxs[:last]])
            yy1 = np.maximum(y1[current_boxes_idx], y1[idxs[:last]])
            xx2 = np.minimum(x2[current_boxes_idx], x2[idxs[:last]])
            yy2 = np.minimum(y2[current_boxes_idx], y2[idxs[:last]])

            width = np.maximum(0, xx2 - xx1 + 1)
            height = np.maximum(0, yy2 - yy1 + 1)

            inter = (width * height)

            union = (areas[current_boxes_idx] + areas[idxs[:last]] - inter)
            overlap = inter / union

            idxs = np.delete(
                idxs,
                np.concatenate(
                    ([last], np.where(overlap >= self.nms_t)[0])
                ))

        return keeped_boxes

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Compute the non-max suppression on the entire image, for each
        detected classes
        :param filtered_boxes: The filtered boxes
        :param box_classes: The corresponding box classes
        :param box_scores: The corresponding score
        :return: The final processed output of the model
        """
        box_predictions = []
        predictions_box_classes = []
        predicted_box_scores = []

        uniques_classes = np.unique(box_classes)

        for class_num in uniques_classes:
            idx_classes = np.where(box_classes == class_num)
            filtered_class_boxes = filtered_boxes[idx_classes]
            filtered_class_box_class = box_classes[idx_classes]
            filtered_class_box_score = box_scores[idx_classes]

            keeped_box_idx = self._nms(
                filtered_class_boxes,
                filtered_class_box_score
            )

            box_predictions.append(filtered_class_boxes[keeped_box_idx])
            predictions_box_classes.append(
                filtered_class_box_class[keeped_box_idx]
            )
            predicted_box_scores.append(
                filtered_class_box_score[keeped_box_idx]
            )

        box_predictions = np.concatenate(box_predictions)
        predictions_box_classes = np.concatenate(predictions_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predictions_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        :param folder_path: The path to the folder that contain pics
        :return: Tuple of two list (image: np.ndarray, images_paths: str)
        """
        output = ([], [])
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                output[0].append(img)
                output[1].append(folder_path + '/' + filename)
        return output

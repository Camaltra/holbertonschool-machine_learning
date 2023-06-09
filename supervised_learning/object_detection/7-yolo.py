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
        image = []
        image_path = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                image.append(img)
                image_path.append(folder_path + '/' + filename)
        return image, image_path

    def preprocess_images(self, images):
        """
        Preocess images -- Put them on the same size as the
        model input and on a scale between [0, 1]
        :param images: The list of images to preprocess
        :return: Tuple of (preprocessed images, original image size)
        """
        images_shapes = []
        images_resized = []
        for image in images:
            images_resized.append(
                cv2.resize(
                    image,
                    dsize=(
                        self.model.input.shape[1].value,
                        self.model.input.shape[2].value
                    ),

                    interpolation=cv2.INTER_CUBIC
                ) / 255
            )
            images_shapes.append(image.shape[:2])

        return np.array(images_resized), np.array(images_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Show box on a given image
        :param image: The image to plot the predictions
        :param boxes: The given predictions
        :param box_classes: The class of predictions
        :param box_scores: The score of predictions
        :param file_name: The file to save to image
        :return: Nothing
        """
        for box_i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)

            cv2.putText(
                image,
                "{} {}".format(
                    self.class_names[box_classes[box_i]],
                    np.around(box_scores[box_i], decimals=2)
                ),
                org=(int(x1), int(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key_pressed = cv2.waitKey(0)
        if key_pressed == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite("detections/{}".format(file_name), image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Make predictions on a given dataset
        :param folder_path: The folder path containing all the images
        :return: The predictions and the images predictions paths
        """
        predictions = []
        images, images_path = self.load_images(folder_path)
        preprocessed_images, original_shape = self.preprocess_images(images)
        model_predictions = self.model.predict(preprocessed_images)
        for idx in range(preprocessed_images.shape[0]):
            output_link_image = [preds[idx] for preds in model_predictions]
            (
                boxes,
                box_confidences,
                box_class_probs
            ) = self.process_outputs(output_link_image, original_shape[idx])
            (
                filtered_boxes,
                box_classes,
                box_scores
            ) = self.filter_boxes(boxes, box_confidences, box_class_probs)
            (
                box_predictions,
                predictions_box_classes,
                predicted_box_scores
            ) = self.non_max_suppression(
                filtered_boxes,
                box_classes,
                box_scores
            )
            predictions.append(
                (
                    box_predictions,
                    predictions_box_classes,
                    predicted_box_scores)
            )
            self.show_boxes(
                images[idx],
                box_predictions,
                predictions_box_classes,
                predicted_box_scores,
                images_path[idx]
            )

        return predictions, images_path

#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf
import os
import numpy as np
from typing import Union, List


def get_classes_name(classes_name_path: str) -> List[str]:
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
            model_path: str,
            classes_name_path: str,
            class_t: float,
            nms_t: float,
            anchors: Union[np.ndarray, List]
    ) -> None:
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

        self.model: tf.keras.Model = tf.keras.models.load_model(model_path)
        self.class_names: List[str] = get_classes_name(classes_name_path)
        self.class_t: float = class_t
        self.nms_t: float = nms_t
        self.anchors: Union[np.ndarray, List] = anchors

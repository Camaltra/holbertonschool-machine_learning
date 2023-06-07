#!/usr/bin/python3

"""Useless comment"""

import tensorflow as tf
import os
import numpy as np
from typing import Union, List


def get_classes_name(classes_name_path: str) -> List[str]:
    classes_name = []
    with open(classes_name_path, "r") as f:
        for line in f.readlines():
            classes_name.append(line.strip())

    return classes_name


class Yolo:
    def __init__(
            self,
            model_path: str,
            classes_name_path: str,
            class_t: float,
            nms_t: float,
            anchors: Union[np.ndarray, List]
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError("Wrong model file path")

        if not os.path.exists(classes_name_path):
            raise FileNotFoundError("Wrong classes file path")

        self.model: tf.keras.Model = tf.keras.models.load_model(model_path)
        self.class_names: List[str] = get_classes_name(classes_name_path)
        self.class_t: float = class_t
        self.nms_t: float = nms_t
        self.anchors: Union[np.ndarray, List] = anchors

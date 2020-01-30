import numpy as np
import cv2

from . import transform


def detect_corner(image):
    return cv2.cornerHarris(image, 2, 3, 0.04)








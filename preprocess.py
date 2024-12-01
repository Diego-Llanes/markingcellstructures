import tifffile
import cv2
import numpy as np

from pathlib import Path
from typing import List, Tuple, Dict
from collections import namedtuple


def normalize(img: np.ndarray):
    '''return the same image normalized between 0-255'''
    return np.uint8(img)

def log_transform(img: np.ndarray):
    return np.log(1 + img)

def gamma_correction(img: np.ndarray, gamma: float=0.5):
    return np.power(img / 255.0, gamma) * 255

def hist_equalize(img: np.ndarray):
    '''contrast limited adaptive histogram equalization'''
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    return clahe.apply(img)

def tophat(img: np.ndarray):
    '''whate tophat transformation'''
    kernel = np.ones((11, 11), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def blur(img: np.ndarray, method: str="gaussian"):
    '''blur the image; choices={gaussian, average, median}'''
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5,5), 0)
    elif method == "median":
        return cv2.medianBlur(img, 5)
    elif method == "average":
        return cv2.blur(img, (5,5))
    else:
        raise NotImplementedError(f"blurring method not implemented: {method}")


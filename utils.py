import torch
import cv2 as cv
import numpy as np

from config import weights

def categorical_crossentropy_color(y_pred, y_true):
    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true.argmax(dim=1))
    cross_ent = torch.mean(cross_ent)

    return cross_ent

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
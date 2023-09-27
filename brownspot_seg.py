import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from kmeans_segmentation import segment_rough
import matplotlib as mpl


def hsvSegOptimised(roughSegImg, mask, threshold_h, threshold_v):
    hsvImage = cv.cvtColor(roughSegImg, cv.COLOR_RGB2HSV)
    h,s,v = cv.split(hsvImage)
    min_t, max_t = threshold_h
    hMask = cv.inRange(h, min_t, max_t)
    vMask = cv.inRange(v, threshold_v[0], threshold_v[1])
    hMask[hMask != 0] = 1
    vMask[vMask != 0] = 1
    # combine hMask and vMask
    fMask = cv.bitwise_and(hMask, vMask)
    mask = cv.bitwise_and(mask, fMask)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=3)
    return mask


def segment(img_path, threshold_h=(1,22),threshold_v=(0,149)):
    roughSegImg, mask = segment_rough(img_path)
    return hsvSegOptimised(roughSegImg, mask, threshold_h, threshold_v)



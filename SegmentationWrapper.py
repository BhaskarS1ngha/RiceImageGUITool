import cv2 as cv
import numpy as np
from brownspot_seg import segment, hsvSegOptimised
from kmeans_segmentation import segment_rough
from scipy.stats import entropy
import os


class ImageSegmenter:
    '''
    This class is a wrapper for the segmentation algorithm that generates a mask for the image at imgSrc.
    :param imgSrc: the path to the image to be segmented
    :param thresholds: a dictionaty that contains the threshold for H and V channels {H: (min, max), V: (min, max)}
    :param label: label to be assigned to the mask. Default is 1
    The generated mask is stored in the mask attribute.
    Contours of the generated mask are stored in the contours attribute.
    segment method generates the mask and contours and stores them in the respective attributes.
    '''

    def __init__(self, imgSrc:str, thresholds:dict,label=1):
        self.imgSrc = imgSrc
        self.h_max = thresholds['H'][1]
        self.h_min = thresholds['H'][0]
        self.v_max = thresholds['V'][1]
        self.v_min = thresholds['V'][0]
        self.label = label

        self.h_absolutes = (0, 180)
        self.v_absolutes = (0, 255)
        self.srcImage = cv.imread(self.imgSrc)
        self.roughSegImg, self.roughMask = segment_rough(self.imgSrc)
        self.mask = None
        self.contours = None
        self.run_checks()


    def update_thresholds(self, thresholds:dict):
        '''
        This method updates the thresholds for H and V channels and segments the image on the updated values.
        :param thresholds: Dictionary containing the new thresholds for H and V channels {H: (min, max), V: (min, max)}
        '''
        self.h_max = thresholds['H'][1]
        self.h_min = thresholds['H'][0]
        self.v_max = thresholds['V'][1]
        self.v_min = thresholds['V'][0]
        self.run_checks()
        self.segment()

    def run_checks(self):
        # check if h_max and h_min are within the range defined by h_absolute and h_min<=h_max
        if self.h_max > self.h_absolutes[1] or self.h_min < self.h_absolutes[0] or self.h_min > self.h_max:
            raise ValueError("Invalid H values")

        # check if v_max and v_min are within the range defined by v_absolute and v_min<=v_max
        if self.v_max > self.v_absolutes[1] or self.v_min < self.v_absolutes[0] or self.v_min > self.v_max:
            raise ValueError("Invalid V values")

        origImage = cv.imread(self.imgSrc)

        # check if image is read
        if origImage is None:
            raise FileNotFoundError("Image not found")

    def segment(self):
        '''
        This method generates a mask for the image at imgSrc and stores it in the mask attribute.
        '''

        mask = hsvSegOptimised(self.roughSegImg, self.roughMask, (self.h_min, self.h_max), (self.v_min, self.v_max))
        mask[mask!=0] = 255
        self.contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mask[mask!=0] = self.label
        self.mask = mask

    def getStats(self):
        '''
        This method returns a dictionary containing the stats of the generated mask.
        '''
        if self.mask is None:
            self.segment()
        stats = {}
        pxidx = np.where(self.mask != 0)
        pxidx = list(zip(pxidx[0], pxidx[1]))
        vals = list()
        for x in pxidx:
            vals.append(self.roughSegImg[x])
        # if len(vals) == 0:
        #     return None
        # dist = list()
        # for i in range(len(vals)):
        #     for j in range(i+1, len(vals)):
        #         dist.append(np.linalg.norm(vals[i]-vals[j]))
        # stats['mean'] = np.mean(dist)
        # stats['std'] = np.std(dist)
        stats['mean_intenisty'] = np.mean(vals)
        stats['std_intensity'] = np.std(vals)
        ent_vals = entropy(vals)
        stats['entropy'] = entropy(vals)
        stats['variance'] = np.var(vals)
        # if len(stats['entropy'])>1:
        #     stats['entropy'] = np.mean(stats['entropy'])
        return stats



if __name__ == '__main__':
    imgSrc = r"Sample Data/IMG_2977.jpg"
    thresholds = {'H': (0, 22), 'V': (0, 149)}
    segmenter = ImageSegmenter(imgSrc, thresholds)
    segmenter.segment()
    st = segmenter.getStats()
    print(st)
    mask = segmenter.mask
    print(mask.shape)
    mask[mask!=0] = 255
    cv.imshow('mask', cv.resize(mask, (800, 800)))
    cv.waitKey(0)
    cv.destroyAllWindows()

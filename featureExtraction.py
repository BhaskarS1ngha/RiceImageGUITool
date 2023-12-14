from __future__ import print_function
import os
import collections
import SimpleITK as sitk
import numpy
import six
import radiomics
from radiomics import firstorder, glcm, imageoperations, glrlm, glszm
from radiomics import shape2D
import cv2 as cv


def check_and_create(path):
    if not os.path.exists(path):
        os.makedirs(path)



settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None
# settings['resampledPixelSpacing'] = [3, 3, 3]  # This is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = 'sitkBSpline'
settings['verbose'] = True
settings['force2D'] = True

def extractor(original_image,original_mask,label=1):
    image = sitk.GetImageFromArray(cv.cvtColor(original_image, cv.COLOR_BGR2GRAY))
    mask = sitk.GetImageFromArray(original_mask)
    print(original_mask.shape)
    check_and_create("temp")
    sitk.WriteImage(image, "temp/image.nrrd")
    sitk.WriteImage(mask, "temp/mask.nrrd")
    del image, mask
    image = sitk.ReadImage("temp/image.nrrd")
    mask = sitk.ReadImage("temp/mask.nrrd")

    # Resample if necessary
    interpolator = settings.get('interpolator')
    resampledPixelSpacing = settings.get('resampledPixelSpacing')
    if interpolator is not None and resampledPixelSpacing is not None:
        image, mask = imageoperations.resampleImage(image, mask, **settings)

    try:
        # Crop the image
        # bb is the bounding box, upon which the image and mask are cropped
        bb, correctedMask = imageoperations.checkMask(image, mask, label=1)
        if correctedMask is not None:
            mask = correctedMask
        croppedImage, croppedMask = imageoperations.cropToTumorMask(image, mask, bb)
    except:
        print("Error in checkMask")
        return None
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(croppedImage, croppedMask, **settings)

    # Set the features to be calculated
    # firstOrderFeatures.enableFeatureByName('Mean', True)
    firstOrderFeatures.enableAllFeatures()
    print("Calculating first order features")

    result = firstOrderFeatures.execute()
    print(result)
    return result




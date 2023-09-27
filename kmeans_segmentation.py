import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import SimpleITK as sitk

CLUSTER_POS = None



def click_event(event, x, y, flags, params):
    global CLUSTER_POS
    if event == cv.EVENT_LBUTTONDOWN:
        CLUSTER_POS = (x, y)
        print(CLUSTER_POS)
        cv.destroyWindow('display1')


# def segmentation(imgPath: str):
#     src = cv.imread(imgPath)
#     roughSegImg, mask = segment_rough(imgPath)
#     if len(mask.shape) >= 3:
#         raise ValueError('mask1 must be a binary image')
#     lmask = labSegmentation(src)
#     mask = cv.bitwise_and(mask, lmask)
#     return roughSegImg, mask


def find_max_cluster(segImageSrc,lmask, centers):
    '''
    function that takes a segmented image, cluster centers, and a mask and returns the mask of the largest cluster
    :param segImage: np.ndarray
    :param lmask: np.ndarray
    :return: mask_final: np.ndarray
    '''
    indx = np.where(lmask !=0)
    max_pix = 0
    segImage = segImageSrc.copy()
    segImage2 = cv.bitwise_and(segImage, segImage, mask=lmask)
    final_center = None
    for center in centers:
        if np.array_equal(center, [0, 0, 0]):
            continue
        layer = segImage2.copy()
        mask = cv.inRange(layer, center, center)
        layer[mask == 0] = [0]
        layer[mask != 0] = [255]
        tmax = np.count_nonzero(layer)
        if tmax >= max_pix:
            final_center = center
            max_pix = tmax


    # print(final_center)
    layer = segImage2.copy()
    mask = cv.inRange(layer, final_center, final_center)
    layer[mask == 0] = [0]
    layer[mask != 0] = [255]
    return layer



def find_smallest_cluster_grayscale(segmnetedImage, centers):
    min_pix = segmnetedImage.shape[0] * segmnetedImage.shape[1]
    mask_final = None
    for center in centers:
        layer = segmnetedImage.copy()
        mask = cv.inRange(layer, center, center)
        layer[mask == 0] = [0]
        layer[mask != 0] = [1]

        tmin = np.count_nonzero(layer)
        if tmin <= min_pix:
            min_pix = tmin
            mask_final = layer.copy()
    return mask_final


def find_smallest_cluster_rgb(segmnetedImage, centers):
    min_pix = segmnetedImage.shape[0] * segmnetedImage.shape[1]
    mask_final = None
    for center in centers:
        layer = segmnetedImage.copy()
        mask = cv.inRange(layer, center, center)
        layer[mask == 0] = [0]
        layer[mask != 0] = [255]

        tmin = np.count_nonzero(layer)
        if tmin <= min_pix:
            min_pix = tmin
            mask_final = layer.copy()
            mask_final = cv.cvtColor(mask_final, cv.COLOR_BGR2GRAY)
    return mask_final


def generate_mask(segmentedImage, center):
    layer = segmentedImage.copy()
    mask = cv.inRange(layer, center, center)
    layer[mask == 0] = [0]
    layer[mask != 0] = [255]
    return layer


def select_cluster(segmentedImage,src):
    global CLUSTER_POS
    resized_image = cv.resize(segmentedImage, (1024,800))
    winName= 'display1'
    CLUSTER_POS = None
    plt.imshow(src)
    plt.show(block=False)
    cv.imshow(winName,resized_image)
    cv.setMouseCallback(winName, click_event)
    cv.waitKey(0)
    plt.close()
    if CLUSTER_POS is None:
        raise Exception("No cluster selected")
    x, y = CLUSTER_POS
    return resized_image[y,x]



# def segment_rough(img_path: str):           # old segmentation code
#     img_bgr = cv.imread(img_path)
#     if img_bgr is None:
#         raise FileNotFoundError
#     img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
#     img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
#     grayscale = img_lab[:, :, 2].copy()
#
#     pixel_vals = grayscale.reshape(grayscale.shape[0] * grayscale.shape[1], 1)
#     pixel_vals = np.float32(pixel_vals)
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.95)
#
#     # then perform k-means clustering with number of clusters defined as 2
#     # also random centres are initially chosen for k-means clustering
#     k = 2
#     retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
#     centers = np.uint8(centers)
#     segmented_data = centers[labels.flatten()]
#
#     # reshape data into the original image dimensions
#     segmented_image = segmented_data.reshape(grayscale.shape)
#     mask = find_smallest_cluster_grayscale(segmented_image, centers)
#     maskedImg = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
#     return maskedImg, mask




def segment_rough(img_path: str):
    img_bgr = cv.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError
    img_rgb = cv.cvtColor(img_bgr,cv.COLOR_BGR2RGB)
    img_lab = cv.cvtColor(img_bgr,cv.COLOR_BGR2HSV)
    # grayscale = cv.cvtColor(img_bgr,cv.COLOR_BGR2GRAY)
    grayscale = img_lab[:, :, 1].copy()

    # normalising
    nm = np.zeros(grayscale.shape)
    nm = cv.normalize(grayscale, nm, 0, 255, cv.NORM_MINMAX)
    grayscale = nm


    pixel_vals = grayscale.reshape(grayscale.shape[0] * grayscale.shape[1], 1)
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.95)

    # then perform k-means clustering with number of clusters defined as 2
    # also random centres are initially chosen for k-means clustering
    k = 2
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(grayscale.shape)
    mask = find_smallest_cluster_grayscale(segmented_image,centers)
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=4)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    # x, y, w, h = cv.boundingRect(mask)
    # mask[y:y + h, x:x + w] = 255
    maskedImg = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
    return maskedImg, mask

def segment_rough2(img_path: str):
    img_bgr = cv.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError
    img_rgb = cv.cvtColor(img_bgr,cv.COLOR_BGR2RGB)
    img_lab = cv.cvtColor(img_bgr,cv.COLOR_BGR2HSV)
    # grayscale = cv.cvtColor(img_bgr,cv.COLOR_BGR2GRAY)
    grayscale = img_lab[:, :, 0].copy()

    # normalising
    nm = np.zeros(grayscale.shape)
    nm = cv.normalize(grayscale, nm, 0, 255, cv.NORM_MINMAX)
    grayscale = nm


    pixel_vals = grayscale.reshape(grayscale.shape[0] * grayscale.shape[1], 1)
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.95)

    # then perform k-means clustering with number of clusters defined as 2
    # also random centres are initially chosen for k-means clustering
    k = 2
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(grayscale.shape)
    mask = find_smallest_cluster_grayscale(segmented_image,centers)
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=4)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    # x, y, w, h = cv.boundingRect(mask)
    # mask[y:y + h, x:x + w] = 255
    maskedImg = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
    return maskedImg, mask
def segment_fine(src, roughmask):
    if len(roughmask.shape) >= 3:
        raise ValueError('mask must be a binary image')
    roughmask[roughmask!=0] = 255
    x, y, w, h =  cv.boundingRect(roughmask)
    img= src.copy()
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 250, 0.95)

    # then perform k-means clustering with number of clusters defined as 3
    # also random centres are initially chosen for k-means clustering
    k = 3
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    print(centers)
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(img.shape)
    cropped_segment = segmented_image[y:y + h, x:x + w]
    center = select_cluster(cropped_segment,src)
    # mask = find_smallest_cluster_rgb(segmented_image, centers)
    mask = generate_mask(segmented_image, center)
    print(np.unique(mask))
    if len(mask.shape) >= 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    print(mask.shape)
    x, y, w, h = cv.boundingRect(mask)
    mask[y:y+h, x:x+w] = 255
    # plt.imshow(mask)
    # plt.show()
    maskedImg = cv.bitwise_and(img, img, mask=mask)
    return maskedImg, mask



def segment_finev2(src, roughmask):
    if len(roughmask.shape) >= 3:
        raise ValueError('mask must be a binary image')
    roughmask[roughmask!=0] = 255
    img= src.copy()
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 250, 0.95)

    # then perform k-means clustering with number of clusters defined as 3
    # also random centres are initially chosen for k-means clustering
    k = 3
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    print(centers)
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(img.shape)
    # plt.imshow(segmented_image)
    # plt.show()
    mask = find_max_cluster(segmented_image, roughmask,centers)
    print(np.unique(mask))
    if len(mask.shape) >= 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    print(mask.shape)
    # plt.imshow(mask)
    # plt.show()
    maskedImg = cv.bitwise_and(img, img, mask=mask)
    return maskedImg, mask



def crop_to_mask(img, mask):
    '''
    crops the image to the bounding box of the mask
    :param img:
    :param mask:
    :return: cropped image
    '''
    if len(mask.shape) >= 3:
        raise ValueError('mask must be a binary image')
    mask[mask!=0] = 255
    x, y, w, h =  cv.boundingRect(mask)
    return img[y:y + h, x:x + w]


# def main():
#     src = cv.imread('test.jpg')
#     img, mask = segment_rough('test.jpg')
#     plt.imshow(img)
#     plt.show()
#     mask = labSegmentation(src)
#     if len(mask.shape) >= 3:
#         raise ValueError('mask1 must be a binary image')
#
#
#     img, mask = segment_finev2(img,mask)
#     plt.imshow(img)
#     plt.show()
#     if len(mask.shape) >= 3:
#         mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
#     # cv.imwrite('testimage_segmented.bmp', mask)
#     # sitk.WriteImage(sitk.GetImageFromArray(mask),'outmask.bmp')



# def main2():
#     img_path = "test.jpg"
#     img, lmask = segmentation(img_path)
#     plt.imshow(img)
#     plt.show()
#     plt.imshow(lmask)
#     plt.show()
#     img,mask = segment_finev2(img,lmask)
#     plt.imshow(img)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main2()

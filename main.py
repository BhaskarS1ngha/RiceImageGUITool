import sys
import cv2 as cv
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage, QPixmap

from UIMainWindow import Ui_MainWindow
from SegmentationWrapper import ImageSegmenter as IS
from featureExtraction import extractor

def check_and_create(path):
    if not os.path.exists(path):
        os.makedirs(path)

class MainWindow:
    def __init__(self,inputcsv,outPath="output/"):
        thresholds = {'H': (0, 22), 'V': (0, 149)}
        self.inputs = pd.read_csv(inputcsv)
        self.imgIndex = 0

        # create output dataframe with columns img_no, type, img_path, processed_status, h_min, h_max, v_min, v_max
        outputcsv = "output.csv"
        if os.path.exists(outputcsv):
            self.output = pd.read_csv(outputcsv)
            # read the last processed image
            last_processed = self.output.iloc[-1]['img_no']
            # set the current image to the next image
            self.imgIndex = last_processed + 1
        else:
            self.output = pd.DataFrame(
                columns=['img_no', 'type', 'label', 'img_path', 'processed_status', 'h_min', 'h_max', 'v_min', 'v_max'])

        self.imgPath = self.inputs.iloc[self.imgIndex]['img_path']
        self.outPath = outPath
        check_and_create(self.outPath)
        self.segmenter = IS(self.imgPath, thresholds)
        self.main_window = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)
        self.h_min = self.ui.slider_h_min.value()
        self.h_max = self.ui.slider_h_max.value()
        self.v_min = self.ui.slider_v_min.value()
        self.v_max = self.ui.slider_v_max.value()
        self.mode = 0  # 0=H channel, 1=S channel, 2=V channel, 3=full colour
        self.origImage = None
        self.mask = None
        self.contours = None
        self.showContours = False
        self.stats_keys = ['mean_intenisty', 'std_intensity', 'entropy', 'variance']
        self.stat_desc = {'mean_intenisty': 'Mean ', 'std_intensity': 'Std Dev ',
                          'entropy': 'Entropy', 'variance': 'Variance'}
        self.validContours = []

        #connect sliders to updateVals function
        self.ui.slider_h_min.valueChanged.connect(self.updateVals)
        self.ui.slider_h_max.valueChanged.connect(self.updateVals)
        self.ui.slider_v_min.valueChanged.connect(self.updateVals)
        self.ui.slider_v_max.valueChanged.connect(self.updateVals)


        #set initial values
        self.ui.slider_h_min.setValue(thresholds['H'][0])
        self.ui.slider_h_max.setValue(thresholds['H'][1])
        self.ui.slider_v_min.setValue(thresholds['V'][0])
        self.ui.slider_v_max.setValue(thresholds['V'][1])

        self.ui.actionCycle_colour_map.triggered.connect(self.updateMode)
        self.ui.actionContours.triggered.connect(self.toggleContours)
        self.ui.generateFeaturesButton.clicked.connect(self.updateStats)
        self.ui.saveButton.clicked.connect(self.saveImage)
        self.ui.nextButton.clicked.connect(self.nextImage)
        self.ui.prevButton.clicked.connect(self.prevImage)
        self.ui.filterButton.clicked.connect(self.filter_contours)


        text = f"#Image: {self.imgIndex}\nType: {self.inputs.iloc[self.imgIndex]['type']}\n"
        self.ui.label_stats.setText(text)

        self.ui.label_original_img.mousePressEvent = self.image_clicked

    def image_clicked(self, event):
        print(f"Mouse clicked {event.x()}, {event.y()}")
        x = event.x()
        y = event.y()
        labelSize = self.ui.label_original_img.size()
        pixMapSize = self.ui.label_original_img.pixmap().size()
        width = pixMapSize.width()
        height = pixMapSize.height()
        x0 = int((labelSize.width() - width) / 2)
        y0 = int((labelSize.height() - height) / 2)


        x_relative = (x - x0)
        y_relative = (y - y0)
        print(f"Mouse clicked relative  {x_relative}, {y_relative}")

        #map the coordinates to the original image
        x_original = int(x_relative * self.origImage.shape[1] / width)
        y_original = int(y_relative * self.origImage.shape[0] / height)

        print(f"Mouse clicked original  {x_original}, {y_original}")

        self.check_point_in_contour((x_original, y_original), self.contours)




    def check_point_in_contour(self, point, contour):
        countours = self.contours
        valid_contours = None
        for contour in countours:
            if cv.pointPolygonTest(contour, point, False) >= 0:
                valid_contours = contour
                break
        if valid_contours is not None:
            self.validContours.append(valid_contours)
            # self.update_mask()
            self.displaySrcImage()


    def filter_contours(self):
        if len(self.validContours) == 0:
            return
        self.contours = self.validContours
        self.validContours = []
        self.update_mask()
        self.displaySrcImage()


    def show(self):
        self.main_window.show()

    def updateVals(self):
        self.h_min = self.ui.slider_h_min.value()
        self.h_max = self.ui.slider_h_max.value()
        self.v_min = self.ui.slider_v_min.value()
        self.v_max = self.ui.slider_v_max.value()

        #check if min is greater than max and reset both to min if so
        if self.h_min > self.h_max:
            self.h_min = self.h_max
            self.ui.slider_h_min.setValue(self.h_min)
        if self.v_min > self.v_max:
            self.v_min = self.v_max
            self.ui.slider_v_min.setValue(self.v_min)

        #print values to console
        print(f'h_min: {self.h_min}; h_max: {self.h_max}; v_min: {self.v_min}; v_max: {self.v_max}')
        self.updateImage()
        # self.updateStats()

    def convertImg(self,img):
        #converts the image to a format that can be displayed by Qt
        print(img.shape)
        if len(img.shape) == 2:
            height, width = img.shape
            channel = 1
        else:
            height, width, channel = img.shape
        bytesPerLine = channel * width
        if len(img.shape) == 2:
            print("hit2")
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        else:
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_BGR888)
        return qImg


    def update_mask(self):
        #updates the mask label
        maskLabelHeight = self.ui.label_mask.height()
        maskLabelWidth = self.ui.label_mask.width()
        self.mask = np.zeros_like(self.segmenter.mask)
        print(f"mouse event mask shape = {self.mask.shape}")
        cv.drawContours(self.mask, self.contours, -1, [255,255,255], -1)
        mask = self.convertImg(self.mask)
        self.ui.label_mask.setPixmap(QPixmap(mask).scaled(maskLabelWidth, maskLabelHeight))

    def updateImage(self):
        #updates the img and mask labels
        self.segmenter.update_thresholds({'H': (self.h_min, self.h_max), 'V': (self.v_min, self.v_max)})
        self.displaySrcImage()
        # self.origImage = self.segmenter.srcImage.copy()
        # if self.mode < 3:
        #     temp = cv.split(cv.cvtColor(self.origImage, cv.COLOR_BGR2HSV))[self.mode]
        #     self.origImage = cv.cvtColor(temp, cv.COLOR_GRAY2BGR)
        self.mask = self.segmenter.mask.copy()
        self.mask[self.mask != 0] = 255
        self.mask = cv.cvtColor(self.mask, cv.COLOR_GRAY2BGR)
        self.contours = self.segmenter.contours
        # ImgLabelHeight = self.ui.label_original_img.height()
        # ImgLabelWidth = self.ui.label_original_img.width()
        maskLabelHeight = self.ui.label_mask.height()
        maskLabelWidth = self.ui.label_mask.width()
        # img = self.convertImg(self.origImage)
        print(f"mask shape = {self.mask.shape}")
        mask = self.convertImg(self.mask)
        # # self.ui.label_original_img.setPixmap(QPixmap(r"Sample Data/IMG_2977.jpg").scaled(ImgLabelWidth, ImgLabelHeight))
        # self.ui.label_original_img.setPixmap(QPixmap(img).scaled(ImgLabelWidth, ImgLabelHeight))
        # self.ui.label_original_img.setPixmap(QPixmap(self.convertImg(self.origImage)).scaled(ImgLabelWidth, ImgLabelHeight))
        self.ui.label_mask.setPixmap(QPixmap(mask).scaled(maskLabelWidth, maskLabelHeight))

    def updateMode(self):
        self.mode = (self.mode+1)%4
        print(self.mode)
        self.displaySrcImage()

    def toggleContours(self):
        self.showContours = not self.showContours
        self.displaySrcImage()

    def displaySrcImage(self):
        if (self.mode < 3):
            temp = cv.split(cv.cvtColor(self.segmenter.srcImage, cv.COLOR_BGR2HSV))[self.mode]
            self.origImage = cv.cvtColor(temp, cv.COLOR_GRAY2BGR)
        else:
            self.origImage = self.segmenter.srcImage.copy()
        ImgLabelHeight = self.ui.label_original_img.height()
        ImgLabelWidth = self.ui.label_original_img.width()
        if self.showContours:
            cv.drawContours(self.origImage, self.contours, -1, (0, 0, 255), 3)

            if len(self.validContours) > 0:
                cv.drawContours(self.origImage, self.validContours, -1, (0, 0, 255), -1)
        img = self.convertImg(self.origImage)
        self.ui.label_original_img.setPixmap(QPixmap(img).scaled(ImgLabelWidth, ImgLabelHeight))
        self.ui.label_original_img.setPixmap(QPixmap(self.convertImg(self.origImage)).scaled(ImgLabelWidth, ImgLabelHeight))

    def updateStats(self):
        # keys = self.stats_keys
        text = f"#Image: {self.imgIndex}\n\n"
        # for key in keys:
        #     text += f"{self.stat_desc[key]}: Wait\n"
        # self.ui.label_stats.setText(text)
        #
        # stats = self.segmenter.getStats()
        # ent = stats['entropy']
        # if ent is not None:
        #     ent = np.mean(ent)
        # stats['entropy'] = ent
        # # if stats is None:
        # #     text = ""
        # #     for key in keys:
        # #         text += f"{self.stat_desc[key]}: NaN\n"
        # #     self.ui.label_stats.setText(text)
        # text=""
        # for key in keys:
        #     text += f"{self.stat_desc[key]}: {stats[key]}\n"
        timg = self.segmenter.srcImage.copy()
        tmask = self.segmenter.mask.copy()
        tmask[tmask != 0] = 1
        stats = extractor(timg,tmask)
        if stats is None:
            text += "Error"
        else:
            for key in stats.keys():
                text += f"{key}: {stats[key]:.{4}}\n"

        self.ui.label_stats.setText(text)

    def saveImage(self):
        outfilename=self.imgPath.split('/')[-1].split('.')[0]
        tempMask = self.mask.copy()
        tempMask[tempMask != 0] = 1
        cv.imwrite(self.outPath+outfilename+'_mask.jpg', self.mask)
        cv.imwrite(self.outPath+outfilename+'_mask.bmp', tempMask)

        # check if the image has already been processed
        if self.imgIndex in self.output['img_no'].values:
            # update the row
            self.output.loc[self.output['img_no'] == self.imgIndex, ['img_no', 'type', 'label', 'img_path', 'processed_status', 'h_min', 'h_max', 'v_min', 'v_max']] = [self.imgIndex, self.inputs.iloc[self.imgIndex]['type'], self.inputs.iloc[self.imgIndex]['label'], self.inputs.iloc[self.imgIndex]['img_path'], 1, self.h_min, self.h_max, self.v_min, self.v_max]
        else:
            # add a new row
            self.output = self.output.append({'img_no': self.imgIndex, 'type': self.inputs.iloc[self.imgIndex]['type'], 'label': self.inputs.iloc[self.imgIndex]['label'], 'img_path': self.inputs.iloc[self.imgIndex]['img_path'], 'processed_status': 1, 'h_min': self.h_min, 'h_max': self.h_max, 'v_min': self.v_min, 'v_max': self.v_max}, ignore_index=True)
        self.output.to_csv("output.csv", index=False)

    def nextImage(self):
        text = "Loading next image..."
        self.ui.label_stats.setText(text)
        thresholds = {'H': (self.h_min, self.h_max), 'V': (self.v_min, self.v_max)}
        if(self.imgIndex == len(self.inputs)-1):
            outText = "Reached end of dataset"
            self.ui.label_stats.setText(outText)
            return
        self.imgIndex += 1
        self.contours= None
        self.imgPath = self.inputs.iloc[self.imgIndex]['img_path']
        self.segmenter = IS(self.imgPath, thresholds)
        text = f"#Image: {self.imgIndex}\nType: {self.inputs.iloc[self.imgIndex]['type']}\n"
        self.ui.label_stats.setText(text)
        self.updateImage()
        print("Img loaded")
    def prevImage(self):
        text = "Loading previous image..."
        self.ui.label_stats.setText(text)
        if (self.imgIndex == 0):
            outText = "Reached start of dataset"
            self.ui.label_stats.setText(outText)
            return
        thresholds = {'H': (self.h_min, self.h_max), 'V': (self.v_min, self.v_max)}
        self.imgIndex -= 1
        self.contours = None
        self.imgPath = self.inputs.iloc[self.imgIndex]['img_path']
        self.segmenter = IS(self.imgPath, thresholds)
        text = f"#Image: {self.imgIndex}\nType: {self.inputs.iloc[self.imgIndex]['type']}\n"
        self.ui.label_stats.setText(text)
        self.updateImage()

if __name__ == '__main__':
    inputcsvFile = "input.csv"
    app = QApplication(sys.argv)
    main_window = MainWindow(inputcsvFile)
    main_window.show()
    sys.exit(app.exec_())


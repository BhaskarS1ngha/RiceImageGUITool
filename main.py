import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage, QPixmap

from UIMainWindow import Ui_MainWindow
from SegmentationWrapper import ImageSegmenter as IS

class MainWindow:
    def __init__(self):
        thresholds = {'H': (0, 22), 'V': (0, 149)}
        self.imgPath = r'Sample Data/IMG_2977.jpg'
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

        # text = ""
        # for key in self.stats_keys:
        #     text += f"{self.stat_desc[key]}: Wait\n"


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
        self.updateStats()

    def convertImg(self,img):
        #converts the image to a format that can be displayed by Qt
        height, width, channel = img.shape
        bytesPerLine = channel * width
        # if(channel == 1):
        #     qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        # else:
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_BGR888)
        return qImg

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
            cv.drawContours(self.origImage, self.contours, -1, (0, 255, 0), 3)
        img = self.convertImg(self.origImage)
        self.ui.label_original_img.setPixmap(QPixmap(img).scaled(ImgLabelWidth, ImgLabelHeight))
        self.ui.label_original_img.setPixmap(QPixmap(self.convertImg(self.origImage)).scaled(ImgLabelWidth, ImgLabelHeight))

    def updateStats(self):
        keys = self.stats_keys
        text = ""
        for key in keys:
            text += f"{self.stat_desc[key]}: Wait\n"
        self.ui.label_stats.setText(text)

        stats = self.segmenter.getStats()
        ent = stats['entropy']
        if ent is not None:
            ent = np.mean(ent)
        stats['entropy'] = ent
        # if stats is None:
        #     text = ""
        #     for key in keys:
        #         text += f"{self.stat_desc[key]}: NaN\n"
        #     self.ui.label_stats.setText(text)
        text=""
        for key in keys:
            text += f"{self.stat_desc[key]}: {stats[key]}\n"
        self.ui.label_stats.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

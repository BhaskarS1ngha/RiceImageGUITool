import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow

from UIMainWindow import Ui_MainWindow


class MainWindow:
    def __init__(self):
        self.main_window = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)
        self.h_min = self.ui.slider_h_min.value()
        self.h_max = self.ui.slider_h_max.value()
        self.v_min = self.ui.slider_v_min.value()
        self.v_max = self.ui.slider_v_max.value()

        #connect sliders to updateVals function
        self.ui.slider_h_min.valueChanged.connect(self.updateVals)
        self.ui.slider_h_max.valueChanged.connect(self.updateVals)
        self.ui.slider_v_min.valueChanged.connect(self.updateVals)
        self.ui.slider_v_max.valueChanged.connect(self.updateVals)


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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

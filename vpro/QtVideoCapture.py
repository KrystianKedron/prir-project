from functools import partial
from time import gmtime, strftime

from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import SIGNAL
import sys
import cv2
import Queue

from Effects import sepia, contrast, blur, laplace, black_white
from backend.therad import NormalThread


class ImgWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        super(ImgWidget, self).__init__(parent)
        self.image = None

    def paintEvent(self, event):

        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

    def setImage(self, image):

        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()


class QtVideoCapture(QtGui.QWidget, uic.loadUiType("ui/video_capture.ui")[0]):

    effect_int = 0
    window_width = 600
    window_height = 480

    start_record = False
    counter = 0

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.ImgWidget = ImgWidget(self.ImgWidget)
        self.timer = QtCore.QTimer(self)

        self.recorder = cv2.VideoWriter('out/record.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        self.connect_signals()

        self.timer.start(1)

        backend.start()

    def connect_signals(self):

        self.connect(self.startButton, SIGNAL("clicked()"), self.start_backend)
        self.connect(self.pushButton,  SIGNAL("clicked()"), partial(self.add_effect, effect_int=1))
        self.connect(self.pushButton_2, SIGNAL("clicked()"), partial(self.add_effect, effect_int=2))
        self.connect(self.pushButton_3, SIGNAL("clicked()"), partial(self.add_effect, effect_int=3))
        self.connect(self.pushButton_4, SIGNAL("clicked()"), partial(self.add_effect, effect_int=4))
        self.connect(self.pushButton_5, SIGNAL("clicked()"), partial(self.add_effect, effect_int=5))

        self.menuBackend.triggered[QtGui.QAction].connect(self.change_backend)

        self.timer.timeout.connect(self.update_frame)

    def change_backend(self, action):

        option_str = action.text()

        if 'BUM' in option_str:

            backend.take_photo(self.effect_int)
            self.add_log("The photo take by %s backend" % option_str.split(' ')[0])

    def add_effect(self, effect_int):

        self.effect_int = effect_int

    def start_backend(self):

        if self.start_record is False:

            # self.startButton.setEnabled(False)
            self.startButton.setText('Save record')
            self.startButton.setStyleSheet('background-color:  rgb(255, 0, 0);'
                                           'color: rgb(255, 255, 255);')
            if self.counter > 0:
                self.recorder = cv2.VideoWriter('out/record%d.avi' % self.counter,
                                                cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            self.start_record = True
        else:

            self.start_record = False
            self.recorder.release()
            self.add_log('Video was save in out dir!')
            self.counter += 1
            self.startButton.setText("Start record")
            self.startButton.setStyleSheet('')

    def update_frame(self):

        if not q.empty():

            frame = q.get()
            img = frame["img"]

            if self.start_record is True:

                self.recorder.write(img)

            img_height, img_width, img_colors = img.shape

            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width

            # configure the effect
            img = self.modify_img(img)

            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

    def modify_img(self, img):

        if self.effect_int == 1:
            return sepia(img)
        elif self.effect_int == 2:
            return laplace(img)
        elif self.effect_int == 3:
            return self.add_blur(img)
        elif self.effect_int == 4:
            return self.add_contrast(img)
        elif self.effect_int == 5:
            return black_white(img)
        else:
            return img

    def add_blur(self, img):

        try:

            width, height = self.parse_width_height(self.lineEdit.text(), self.lineEdit_2.text())

            if width != 0 and height != 0:

                return blur(img, width, height)
            else:

                return img
        except ValueError:

            QtGui.QMessageBox.critical(None,
                                       "Error parsing arguments",
                                       "Please put the numeral value only!",
                                       QtGui.QMessageBox.Ok)
            self.effect_int = 0
            return img

    def add_contrast(self, img):

        try:

            width, height = self.parse_width_height(self.lineEdit_3.text(), self.lineEdit_4.text())
            power = self.horizontalSlider.value()

            if width != 0 and height != 0:

                return contrast(img, width, height, power)
            else:

                return img
        except ValueError:

            QtGui.QMessageBox.critical(None,
                                       "Error parsing arguments",
                                       "Please put the numeral value only!",
                                       QtGui.QMessageBox.Ok)
            self.effect_int = 0
            return img

    def closeEvent(self, event):

        backend.stop()

    @staticmethod
    def parse_width_height(text_1, text_2):

        return int(text_1) if text_1 != '' else 0, \
               int(text_2) if text_2 != '' else 0

    def add_log(self, text):

        time = strftime("%H:%M:%S", gmtime())
        log = QtGui.QListWidgetItem("%s: %s" % (time, text))
        self.listWidget.addItem(log)


if __name__ == '__main__':

    q = Queue.Queue()
    backend = NormalThread(0, q, 640, 480, 30)

    app = QtGui.QApplication(sys.argv)
    w = QtVideoCapture(None)
    w.setWindowTitle('OpenCv Video Capture App')
    w.show()
    app.exec_()

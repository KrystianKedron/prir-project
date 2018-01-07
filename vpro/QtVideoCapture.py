from PyQt4 import QtCore, QtGui, uic
import sys
import cv2
import Queue


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

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.startButton.clicked.connect(self.start_clicked)

        self.ImgWidget = ImgWidget(self.ImgWidget)

        self.window_width = 600
        self.window_height = 480

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def start_clicked(self):

        backend.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')

    def update_frame(self):

        if not q.empty():

            self.startButton.setText('Camera is live')
            frame = q.get()
            img = frame["img"]

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

            from Effects import sepia
            img = sepia(img)

            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

    def closeEvent(self, event):

        backend.stop()


if __name__ == '__main__':

    q = Queue.Queue()
    
    from backend.therad import NormalThread
    backend = NormalThread(0, q, 1920, 1080, 30)

    app = QtGui.QApplication(sys.argv)
    w = QtVideoCapture(None)
    w.setWindowTitle('OpenCv Video Capture App')
    w.show()
    app.exec_()
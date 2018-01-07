import threading
import cv2


class NormalThread(threading.Thread):

    _is_running = False

    def __init__(self, cam, queue, width, height, fps):

        super(NormalThread, self).__init__(group=None, target=None, name=None, verbose=None)
        self.capture = cv2.VideoCapture(cam)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

        self.queue = queue

    def run(self):

        self._is_running = True

        while self._is_running:

            frame = {}
            self.capture.grab()
            _, img = self.capture.retrieve(0)
            frame["img"] = img

            if self.queue.qsize() < 10:

                self.queue.put(frame)
            else:
                print "Queue is full!"

    def stop(self):
        self._is_running = False

import threading
import cv2

from prir.vpro.Effects import black_white, contrast, blur, sepia, laplace


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

    def take_photo(self, effect_int):

        _, frame = self.capture.read()
        effect = self.select_effect(effect_int)
        cv2.imwrite("photo_effect.png", effect(frame))

    def select_effect(self, arg):

        if arg == 1:
            return sepia
        elif arg == 2:
            return laplace
        elif arg == 3:
            return blur
        elif arg == 4:
            return contrast
        elif arg == 5:
            return black_white
        else:
            return self.no_effect

    @staticmethod
    def no_effect(img):

        return img
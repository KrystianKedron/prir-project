import cv2
from prir.vpro.recorder.effects import black_white


class VideoCapture:

    def __init__(self):

        self.video = cv2.VideoCapture(0)
        self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    def run(self):

        while True:
            _, frame = self.video.read()
            frame = black_white(frame)
            self.out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    videoCap = VideoCapture()
    videoCap.run()

from mpi4py import MPI

import cv2
import Queue

_is_running = True


def mpi(capture, queue):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    frame = {}
    capture.grab()
    _, img = capture.retrieve(0)

    frame["img"] = img

    if queue.qsize() < 10:

        queue.put(frame)
    else:
        print "Queue is full!"

    print ('Process %d' % rank)


def stop():

    global _is_running
    _is_running = False


def run(cam, queue, width, height, fps):

    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while _is_running:

        mpi(capture, queue)
        if queue.qsize() > 9:
            stop()


if __name__ == '__main__':

    # start()
    q = Queue.Queue()
    run(0, q, 1920, 1080, 30)

    frame = q.get()
    img = frame["img"]

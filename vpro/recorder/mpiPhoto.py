from mpi4py import MPI
import numpy as np
import cv2

from prir.vpro.Effects import sepia, blur, contrast, laplace, black_white


def no_effect(img):

    return img


def select_effect(arg):

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
        return no_effect


def mpi_take_photo(effect_int, video):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    _, frame = video.read()

    recvbuf = None
    if rank == 0:

        recvbuf = np.ndarray(shape=(480, 640, 3), dtype=np.uint8)

    comm.Gather(frame, recvbuf, root=0)

    if rank == 0:
        effect = select_effect(effect_int)
        cv2.imwrite("photo_effect.png", effect(recvbuf))

    video.release()


if __name__ == '__main__':

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    mpi_take_photo(5, capture)
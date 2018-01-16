import cv2
import numpy as np


def sepia(frame):

    return cv2.transform(frame, np.array([[0.189, 0.769, 0.393],
                                          [0.168, 0.686, 0.349],
                                          [0.131, 0.534, 0.272]]))


def laplace(frame):

    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return cv2.filter2D(frame, -1, kernel)


def blur(frame, width=10, height=10):

    return cv2.blur(frame, (width, height))


def black_white(frame):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw_frame = cv2.threshold(frame_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return bw_frame


def contrast(frame, width=8, height=8, power=3.0):

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=power, tileGridSize=(width, height))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def test(filename):

    img = cv2.imread(filename, 1)
    cv2.imshow('sepia', sepia(img))
    cv2.imshow('laplace', laplace(img))
    cv2.imshow('blur', blur(img))
    cv2.imshow('black&white', black_white(img))
    cv2.imshow('contrast', contrast(img))
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    import sys

    if len(sys.argv) == 2:

        test(sys.argv[1])
    else:

        print 'Run script with filename of the image e.q \'python effect.py filename\' !!!'

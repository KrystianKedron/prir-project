from mpi4py import MPI
import numpy as np
import cv2
from timeit import default_timer as timer

from effects import contrast
from utils import read_video_from_file_mpi


def add_effect(_data, _start, _stop):

    tmp = [0 for x in range(len(_data))]
    for i in range(_start, _stop):
        tmp[i] = contrast(_data[i])

    return tmp[_start:_stop]


def mpi_add_effect():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # first process to read video from file
    if rank == 0:

        data, h = read_video_from_file_mpi('record.avi')
        print 'Liczba klatek ', h
    else:
        h = None

    # broadcast to get length of move and allocate array on other ranks:
    h = comm.bcast(h, root=0)
    if rank != 0:
        # allocate the memory to
        data = {}
        for key in range(h):
            data[key] = np.ndarray(shape=(480, 640, 3), dtype=np.uint8)

    data = comm.bcast(data, root=0)

    start = (h / size) * rank
    stop = (h / size) * (rank + 1)
    print 'W procesie %d przetwarzam od %d do %d' % (rank, start, stop)

    mpi_process = add_effect(data, start, stop)
    modify_video = comm.reduce(mpi_process, op=MPI.SUM, root=0)

    if rank == 0:

        dt = timer() - start
        print "Processing done in %f s" % dt
        for i in range(h - 2):
            cv2.imshow('Modify video', modify_video[i])
            cv2.waitKey(50)


if __name__ == '__main__':
    start = timer()
    mpi_add_effect()

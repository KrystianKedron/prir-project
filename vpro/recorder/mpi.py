from mpi4py import MPI
import numpy as np
import cv2

from effects import contrast


def add_effect(_data, _start, _stop):

    tmp = [0 for x in range(len(_data))]
    for i in range(_start, _stop):
        tmp[i] = contrast(_data[i])

    return tmp[_start:_stop]


def read_video_from_file(file_name):

    cap = cv2.VideoCapture(file_name)

    # Check if camera opened successfully
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    video_tab = {}
    video_length = 0
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()

        video_tab[video_length] = frame
        video_length += 1

        if ret is True:

            # Display the resulting frame
            cv2.imshow('Frame', video_tab[video_length - 1])
            # recorder.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # -1 because last frame is disrupted
    return video_tab, video_length - 1


def mpi_add_effect():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # first process to read video from file
    if rank == 0:

        data, h = read_video_from_file('record.avi')
    else:
        h = None

    # broadcast to get length of move and allocate array on other ranks:
    h = comm.bcast(h, root=0)

    if rank != 0:
        # allocate the memory to data
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
        for i in range(h - 1):
            cv2.imshow('Modify video', modify_video[i])
            cv2.waitKey(50)


if __name__ == '__main__':

    mpi_add_effect()

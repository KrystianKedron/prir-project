import cv2
import numpy as np


def read_video_from_file_cuda(file_name):

    cap = cv2.VideoCapture(file_name)

    # Check if camera opened successfully
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    video_tab = []
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret is True:
            video_tab.append(frame.tolist())
            # Display the resulting frame
            # cv2.imshow('Frame', video_tab[video_length - 1])
            pass
            # recorder.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # -1 because last frame is disrupted
    return np.asarray(video_tab)


def read_video_from_file_mpi(file_name):

    cap = cv2.VideoCapture(file_name)

    # Check if camera opened successfully
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    video_tab = {}
    video_length = 0
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret is True:

            video_tab[video_length] = frame
            video_length += 1
            # Display the resulting frame
            # cv2.imshow('Frame', video_tab[video_length - 1])
            pass
            # recorder.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # -1 because last frame is disrupted
    return video_tab, video_length - 1
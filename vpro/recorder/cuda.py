from numba import cuda
from timeit import default_timer as timer

import cv2
# print(cv2.getBuildInformation())
from utils import read_video_from_file_cuda


# @cuda.jit(device=True)
# def blur(frame):
#
#     return cv2.blur(frame, (10, 10))


@cuda.jit()
def procces_video(video):

    index = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    video[index] = 255


N = 108
M = 9
gvideo = read_video_from_file_cuda('record.avi')
start = timer()

d_video = cuda.to_device(gvideo)

procces_video[(N+M-1)/M, M](d_video)
d_video.to_host()

dt = timer() - start

print "Processing in GPU done in %f s" % dt
# print 'Modify video ', gimage
cv2.imshow('New video', d_video[0])
cv2.waitKey(0)

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
def mandel_kernel(image):

    index = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    image[index] = 255


N = 108
M = 9
gimage = read_video_from_file_cuda('filimk300.avi')
start = timer()

d_image = cuda.to_device(gimage)

mandel_kernel[(N+M-1)/M, M](d_image)
d_image.to_host()

dt = timer() - start

print "Mandelbrot created on GPU in %f s" % dt
# print 'Modify video ', gimage
cv2.imshow('New video', gimage[0])
cv2.waitKey(0)
#!/usr/bin/env python

'''
Advanced Python assignemtn 12:
    mandelbrot_gpu.py
'''


# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
        1. get the imaege shape
        2. figure out the settings of kernel launch 
        3. get the thread indices in each dimension 
        4. calculate out how many pixels should be handled by each thread
        5. nested loop to call mandel() and assign value to each pixels
    '''
    # get parameters of image settings 
    height = image.shape[0]
    width = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    
    # total threads on each axis
    grid_tt_x = cuda.blockDim.x * cuda.gridDim.x
    grid_tt_y = cuda.blockDim.y * cuda.gridDim.y
    
    # number of pixels assigned to each thread
    per_x = (width + grid_tt_x - 1) // grid_tt_x  
    per_y = (height + grid_tt_y - 1) // grid_tt_y
    
    # current thread idx on each dimension
    idx_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    idx_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    
    for offset_x in range(per_x):
        px_idx_x = idx_x * per_x + offset_x  # current pixel idx on X
        real = min_x + px_idx_x * pixel_size_x 
        for offset_y in range(per_y):
            px_idx_y = idx_y * per_y + offset_y  # current pixel idx on Y
            imag = min_y + px_idx_y * pixel_size_y
            if (px_idx_x < width and px_idx_y < height):
                image[px_idx_y, px_idx_x] = mandel(real, imag, iters)
        
    
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8) # 1572864 pixels
    blockdim = (32, 8)  # 32 * 8 thread block,  256 threads
    griddim = (32, 16)  # 32 * 16 block grid, 512 blocks * 256 = 131072 => 12 pixel per thread
    # 32 * 32 = 1024 in X-axis -> each treahd should take care at most 2 pixel on X-axis
    # 8 * 16 = 128 in Y-axis -> each thread should take care 1024/128 = 8 pixel on Y-axis
    
    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()
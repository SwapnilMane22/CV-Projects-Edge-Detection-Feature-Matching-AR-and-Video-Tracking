import numpy as np
from scipy import signal    # For signal.gaussian function

from myImageFilter import myImageFilter
import math

def myEdgeFilter(img0, sigma):
    # YOUR CODE HERE
    hsize = 2 * math.ceil(3 * sigma) + 1  
    gaussian_filter = create_gaussian_filter(hsize, sigma)

    smoothed_img = myImageFilter(img0, gaussian_filter)

    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1], 
                        [ 0,  0,  0], 
                        [ 1,  2,  1]])

    imgx = myImageFilter(smoothed_img, sobel_x)
    imgy = myImageFilter(smoothed_img, sobel_y)

    gradient_magnitude = np.hypot(imgx, imgy)
    gradient_direction = np.arctan2(imgy, imgx) * 180 / np.pi  

    non_max_suppressed_img = non_maximum_suppression(gradient_magnitude, gradient_direction)

    return non_max_suppressed_img

def non_maximum_suppression(magnitude, direction):
    direction = (direction + 180) % 180

    img_height, img_width = magnitude.shape
    output_img = np.zeros((img_height, img_width), dtype=np.float32)

    magnitude_padded = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant')

    angle = np.zeros_like(direction)
    angle[(0 <= direction) & (direction < 22.5)] = 0
    angle[(157.5 <= direction) & (direction <= 180)] = 0
    angle[(22.5 <= direction) & (direction < 67.5)] = 45
    angle[(67.5 <= direction) & (direction < 112.5)] = 90
    angle[(112.5 <= direction) & (direction < 157.5)] = 135

    for theta in [0, 45, 90, 135]:
        if theta == 0:
            neighbors1 = magnitude_padded[1:-1, 2:]  
            neighbors2 = magnitude_padded[1:-1, :-2] 
        elif theta == 45:
            neighbors1 = magnitude_padded[:-2, 2:]
            neighbors2 = magnitude_padded[2:, :-2] 
        elif theta == 90:
            neighbors1 = magnitude_padded[:-2, 1:-1] 
            neighbors2 = magnitude_padded[2:, 1:-1]  
        elif theta == 135:
            neighbors1 = magnitude_padded[:-2, :-2] 
            neighbors2 = magnitude_padded[2:, 2:] 

        mask = (angle == theta) & (magnitude >= neighbors1) & (magnitude >= neighbors2)
        output_img[mask] = magnitude[mask]

    return output_img

def create_gaussian_filter(hsize, sigma):
    hsize = 2 * int(np.ceil(3 * sigma)) + 1 

    gaussian_filter = np.zeros((hsize, hsize), dtype=np.float32)
    
    center = hsize // 2
    
    for i in range(hsize):
        for j in range(hsize):
            x = i - center
            y = j - center
            gaussian_filter[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    gaussian_filter /= np.sum(gaussian_filter)
    
    return gaussian_filter
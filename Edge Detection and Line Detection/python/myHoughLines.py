import numpy as np
# import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    # YOUR CODE HERE
    neighborhood_size = 3
    half_size = neighborhood_size // 2
    
    padded_H = np.pad(H, pad_width=half_size, mode='constant', constant_values=0)
    
    suppressed_H = H.copy()
    
    for i in range(half_size, H.shape[0] + half_size):
        for j in range(half_size, H.shape[1] + half_size):
            neighborhood = padded_H[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            
            max_value = np.max(neighborhood)
            
            if padded_H[i, j] != max_value:
                suppressed_H[i - half_size, j - half_size] = 0

    flat_indices = np.argsort(suppressed_H.ravel())[::-1]
    top_indices = flat_indices[:nLines]
    
    rho_indices, theta_indices = np.unravel_index(top_indices, H.shape)
    
    return rho_indices, theta_indices
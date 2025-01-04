import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    # YOUR CODE HERE

    img_height, img_width = Im.shape
    
    diagonal_length = np.sqrt(img_height**2 + img_width**2)
    rho_max = int(np.ceil(diagonal_length))
    
    rhoScale = np.arange(-rho_max, rho_max + rhoRes, rhoRes)
    thetaScale = np.arange(0, np.pi, thetaRes)

    hough_accumulator = np.zeros((len(rhoScale), len(thetaScale)), dtype=np.int32)
    
    y_idxs, x_idxs = np.nonzero(Im > 0)
    
    theta_values = thetaScale[np.newaxis, :]
    
    x_coords = x_idxs[:, np.newaxis]
    y_coords = y_idxs[:, np.newaxis]
    
    rho_values = x_coords * np.cos(theta_values) + y_coords * np.sin(theta_values)
    
    rho_indices = np.round((rho_values - rhoScale[0]) / rhoRes).astype(int)
    
    valid_mask = (rho_indices >= 0) & (rho_indices < len(rhoScale))
    
    rho_indices = rho_indices[valid_mask]
    
    theta_indices = np.tile(np.arange(len(thetaScale)), len(x_idxs)).reshape(len(x_idxs), len(thetaScale))[valid_mask]
    
    np.add.at(hough_accumulator, (rho_indices, theta_indices), 1)
    
    return hough_accumulator, rhoScale, thetaScale
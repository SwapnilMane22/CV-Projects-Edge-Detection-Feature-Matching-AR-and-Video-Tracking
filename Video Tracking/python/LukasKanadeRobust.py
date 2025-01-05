import numpy as np
from scipy.interpolate import RectBivariateSpline

# Get weight matrix for M-Estimator
# Default a given by https://www.real-statistics.com/descriptive-statistics/m-estimators/
def getWeightMatrix(e, a=None, mtype="huber", diag_matrix=False):
    if mtype == "huber":
        if a is None:
            a = 1.339
        weights = np.tile(a, e.shape[0]) / np.abs(e)
    elif mtype == "tukey":
        if a is None:
            a = 4.685
        weights = (np.ones_like(e) - (e/a)**2)**3
    elif mtype == "none":
        weights = np.ones_like(e)

    np.clip(weights, 0, 1, out=weights)

    if diag_matrix:
        return np.diag(weights)

    return weights


def LucasKanadeAffineRobust(It, It1, rect, mtype):
    """
    Robust Lucas-Kanade tracker with affine motion model.
    
    Parameters:
    - It: Template image
    - It1: Current image
    - rect: Current position of the object (top left, bottom right coordinates: x1, y1, x2, y2)
    - mtype: Type of M-estimator to use ("huber", "tukey", "none")
    
    Returns:
    - p: movement vector dx, dy (or affine parameters)
    """

    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64
    
    p = np.zeros(6, dtype=npDtype)  # Affine parameters (dx, dy, and affine terms)
    x1, y1, x2, y2 = rect
    
    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)
    
    # Create spline interpolators for both images
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    nX, nY = int(x2 - x1), int(y2 - y1)
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)
    xx, yy = np.meshgrid(coordsX, coordsY)

    template = splineT.ev(xx, yy)

    # Begin iterative refinement
    for _ in range(maxIters):

        # Warp coordinates for the affine transformation
        M = np.array([[1 + p[0], p[1], p[2]],
                      [p[3], 1 + p[4], p[5]]])
        coords = np.vstack((xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())))
        warped_coords = M @ coords
        xx_prime = warped_coords[0, :].reshape(xx.shape)
        yy_prime = warped_coords[1, :].reshape(yy.shape)

        # Clamp coordinates to valid bounds
        xx_prime = np.clip(xx_prime, 0, width - 1)
        yy_prime = np.clip(yy_prime, 0, height - 1)

        # Get the warped image from the new coordinates
        warpedI = splineI.ev(xx_prime, yy_prime)

        # Compute error image
        error = template - warpedI

        # Get weights using M-estimator (robust error handling)
        weights = getWeightMatrix(error, mtype=mtype)
        
        # Ensure weights have correct shape (14400,)
        weights = weights.flatten()  # Flatten to match the gradient shape
        
        # Compute gradient of the warped image
        gradX = splineI.ev(xx_prime, yy_prime, dx=1, dy=0)
        gradY = splineI.ev(xx_prime, yy_prime, dx=0, dy=1)
        
        # Stack gradients into a Jacobian matrix
        grad = np.stack((gradX.ravel(), gradY.ravel()), axis=1)
        
        # Apply weights element-wise to gradients
        weighted_grad = grad * weights[:, np.newaxis]  # Now this works with broadcasting
        
        # Compute Hessian (weighted)
        H = weighted_grad.T @ grad
        
        # Compute weighted error vector
        b = weighted_grad.T @ (weights * error.ravel())
        
        # Solve for deltaP
        deltaP = np.linalg.lstsq(H, b, rcond=None)[0]
        
        # Ensure deltaP is of size (6,) if not already
        if deltaP.shape[0] < 6:
            deltaP = np.pad(deltaP, (0, 6 - deltaP.shape[0]), mode='constant')

        # Update parameters
        p += deltaP

        # Break if update is smaller than the threshold
        if np.linalg.norm(deltaP) < threshold:
            break

    return p
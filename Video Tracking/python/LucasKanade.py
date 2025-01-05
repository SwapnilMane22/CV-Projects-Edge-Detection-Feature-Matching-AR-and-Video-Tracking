import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

def LucasKanade(It, It1, rect):
    """
    Q3.1
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] p: movement vector dx, dy
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64    # Might be useful
    # p := dx, dy
    p = np.zeros(2, dtype=npDtype)  # OR p = np.zeros(2)
    x1, y1, x2, y2 = rect

    # Crop template image
    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)

    # This returns a class object; note the swap/transpose
    # Use spline.ev() for getting values at locations
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    nX, nY = int(x2 - x1), int(y2 - y1)
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)
    xx, yy = np.meshgrid(coordsX, coordsY)

    # YOUR IMPLEMENTATION STARTS HERE

    template = splineT.ev(xx, yy)

    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        # Warp image
        #   1. warp coordinates
        xx_prime = xx + p[0]
        yy_prime = yy + p[1]

        # Clamp coordinates to valid bounds
        xx_prime = np.clip(xx_prime, 0, width - 1)
        yy_prime = np.clip(yy_prime, 0, height - 1)

        #   2. warp image (get image from new image locations)
        warpedI = splineI.ev(xx_prime, yy_prime)

        # Compute error image
        error = template - warpedI 

        # Compute gradient of warped image
        gradX = splineI.ev(xx_prime, yy_prime, dx=1, dy=0)
        gradY = splineI.ev(xx_prime, yy_prime, dx=0, dy=1)

        # Stack gradients into a Jacobian matrix
        grad = np.stack((gradX.ravel(), gradY.ravel()), axis=1)

        # Compute Hessian; It is a special case
        H = grad.T @ grad

        # Calculate deltaP
        deltaP = np.linalg.lstsq(H, grad.T @ error.ravel(), rcond=None)[0]

        # Update p
        p += deltaP

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break

    return p[0], p[1] #p

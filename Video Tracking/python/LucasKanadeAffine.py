import numpy as np
from scipy.interpolate import RectBivariateSpline
import math

def LucasKanadeAffine(It, It1, rect):
    """
    Q3.2
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] M: the Affine warp matrix [2x3 numpy array]
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64    # Might be helpful
    p = np.zeros((6, 1), dtype=npDtype) # OR p = np.zeros((6,1))
    x1, y1, x2, y2 = rect

    # Get image dimensions
    height, width = It1.shape
    _x, _y = np.arange(width), np.arange(height)

    # Create meshgrid for the template
    x1 = np.clip(x1, 0, width - 1)
    x2 = np.clip(x2, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    y2 = np.clip(y2, 0, height - 1)

    # nX, nY = int(abs(x2 - x1)), int(abs(y2 - y1))
    nX, nY = int(max(8, abs(x2 - x1))), int(max(4, abs(y2 - y1)))
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)
    # coordsX = np.clip(coordsX, 0, width - 1)
    # coordsY = np.clip(coordsY, 0, height - 1)
    xx, yy = np.meshgrid(coordsX, coordsY)
    # x = np.arange(x1, x2 + nX)
    # y = np.arange(y1, y2 + nY)
    # xx, yy = np.meshgrid(x, y)

    # Interpolate the template and current image
    splineT = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    splineI = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    # splineT = RectBivariateSpline(_x, _y, It.T)
    # splineI = RectBivariateSpline(_x, _y, It1.T)

    # Template intensity values
    T = splineT.ev(xx, yy)

    # YOUR IMPLEMENTATION HERE

    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        # Warp image
        #   1. warp coordinates
        M = np.array([[1 + p[0, 0], p[1, 0], p[2, 0]],
                      [p[3, 0], 1 + p[4, 0], p[5, 0]]])

        coords = np.vstack((xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())))
        warped_coords = M @ coords
        xx_prime = warped_coords[0, :].reshape(xx.shape)
        yy_prime = warped_coords[1, :].reshape(yy.shape)

        # Clamp coordinates to valid bounds
        xx_prime = np.clip(xx_prime, 0, width - 1)
        yy_prime = np.clip(yy_prime, 0, height - 1)

        #   2. warp image (get image from new image locations)
        warpedI = splineI.ev(xx_prime, yy_prime)

        # Compute error image
        error = T - warpedI

        # Compute gradient of warped image
        Ix = splineI.ev(xx_prime, yy_prime, dx=1, dy=0).ravel()
        Iy = splineI.ev(xx_prime, yy_prime, dx=0, dy=1).ravel()

        # Compute Jacobian and Hessian
        gradI = np.vstack((Ix, Iy)).T
        J = np.zeros((len(Ix), 6), dtype=npDtype)
        J[:, 0] = gradI[:, 0] * xx.ravel()
        J[:, 1] = gradI[:, 0] * yy.ravel()
        J[:, 2] = gradI[:, 0]
        J[:, 3] = gradI[:, 1] * xx.ravel()
        J[:, 4] = gradI[:, 1] * yy.ravel()
        J[:, 5] = gradI[:, 1]

        # Calculate deltaP
        H = J.T @ J

        # Regularization: check if H is ill-conditioned and add regularization if necessary
        # det_H = np.linalg.det(H)
        # if det_H < 1e-6:
        #     # Regularize the Hessian (adding a small value to the diagonal)
        #     H += 1e-6 * np.eye(6) #H.shape[0]
        #     print(f"Regularized Hessian due to poor conditioning (det_H: {det_H})")

        # deltaP = np.linalg.inv(H) @ (J.T @ error.ravel())
        deltaP = np.linalg.lstsq(H, J.T @ error.ravel(), rcond=None)[0]

        # Update p
        p += deltaP.reshape(6, 1)

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break


    # Reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)
    # M = np.array([[1 + p[0, 0], p[1, 0], p[2, 0]],
    #               [p[3, 0], 1 + p[4, 0], p[5, 0]]]).reshape(2, 3)

    return M

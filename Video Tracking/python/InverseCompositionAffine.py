import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    """
    Q3.3
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] M: the Affine warp matrix [2x3 numpy array]
    """

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    # p = np.zeros((6,1))
    npDtype = np.float64
    W = np.eye(3, dtype=npDtype)    # This might be a better format than p
    x1, y1, x2, y2 = rect

    # YOUR IMPLEMENTATION HERE   

    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height) 

    # Compute gradient of template image
    splineT = RectBivariateSpline(_y, _x, It)

    # Define grid for the template region

    coordsX = np.linspace(x1, x2, int(abs(x2 - x1)), dtype=npDtype)
    coordsY = np.linspace(y1, y2, int(abs(y2 - y1)), dtype=npDtype)
    xx, yy = np.meshgrid(coordsX, coordsY)
    template = splineT.ev(yy, xx)

    # Compute Jacobian

    grad_x = splineT.ev(yy, xx, dx=1)
    grad_y = splineT.ev(yy, xx, dy=1)

    J = np.zeros((len(grad_x.ravel()), 6))
    J[:, 0] = grad_x.ravel() * xx.ravel()
    J[:, 1] = grad_x.ravel() * yy.ravel()
    J[:, 2] = grad_x.ravel()
    J[:, 3] = grad_y.ravel() * xx.ravel()
    J[:, 4] = grad_y.ravel() * yy.ravel()
    J[:, 5] = grad_y.ravel()

    # Compute Hessian

    H = J.T @ J  # Precompute Hessian matrix
    splineI = RectBivariateSpline(_y, _x, It1)

    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        # Warp image
        warped_x = W[0, 0] * xx + W[0, 1] * yy + W[0, 2]
        warped_y = W[1, 0] * xx + W[1, 1] * yy + W[1, 2]

        warped_image = splineI.ev(warped_y, warped_x)

        # Compute error image
        error = template - warped_image

        # Compute deltaP
        b = J.T @ error.ravel()
        deltaP = np.linalg.pinv(H) @ b

        # Compute new W
        dp_mat = np.array([[1 + deltaP[0], deltaP[1], deltaP[2]],
                           [deltaP[3], 1 + deltaP[4], deltaP[5]],
                           [0, 0, 1]])
        # W = np.vstack([W, [0, 0, 1]])
        W = np.linalg.inv(dp_mat) @ W

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break


    # reshape the output affine matrix
    # M = np.array([[1.0+p[0], p[1],    p[2]],
    #              [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)
    M = W[:2, :]

    return M

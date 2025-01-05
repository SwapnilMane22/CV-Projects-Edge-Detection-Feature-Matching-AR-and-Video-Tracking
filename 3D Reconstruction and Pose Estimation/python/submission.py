"""
Programming Assignment 3
Submission Functions
"""

# import packages here
import helper
import numpy as np
import cv2
from scipy.signal import convolve2d
import scipy

"""
Q2.1 Eight Point Algorithm
   [I] pts1 -- points in image 1 (Nx2 matrix)
       pts2 -- points in image 2 (Nx2 matrix)
       M -- scalar value computed as max(H, W)
   [O] F -- the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2):
    # Normalize the points
    pts1, t1 = helper.normalize_points(pts1)
    pts2, t2 = helper.normalize_points(pts2)

    # Formulate the A matrix
    N = pts1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Calculate F using SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # Refine the solution by using local minimization
    F = helper.refineF(F, pts1, pts2)

    # Un-normalize F
    F = t2.T @ F @ t1

    # Return the fundamental matrix
    return F

    # Replace pass with output
    # pass


"""
Q2.2 Epipolar Correspondences
   [I] im1 -- image 1 (H1xW1 matrix)
       im2 -- image 2 (H2xW2 matrix)
       F -- fundamental matrix from image 1 to image 2 (3x3 matrix)
       pts1 -- points in image 1 (Nx2 matrix)
   [O] pts2 -- points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):


    # Note: pts1 is in heterogeneous coordinate system
    #   Converts pts1 to homogeneous coordinate system
    H, W, _ = im2.shape
    # H, W = im2.shape
    if pts1.ndim == 1:
      pts1 = pts1.reshape(1, -1)

    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1)))) 

    # Note: epipolar lines (ax + by + c = 0)
    window_size = 5
    half_window = window_size // 2

    pts2 = []

    # Calculate candidate corresponding points along epipolar line in im2

    # Create image patch(es)
    # Remember to pad your image to get your patches.
    # This is a nifty function to create multiple patches in one line:
    #   np.lib.stride_tricks.sliding_window_view(padded_im, window_shape=(width, width), axis=(0, 1))
   
    for pt1 in pts1_h:
        # Compute the epipolar line l' in im2
        l_prime = F @ pt1.T
        a, b, c = l_prime
        # epipolar_line = np.dot(F, pt1)

        # Normalize the line equation
        # a, b, c = epipolar_line / np.linalg.norm(epipolar_line[:2])

        norm_factor = np.sqrt(a**2 + b**2)
        if norm_factor == 0:
            continue
        l_prime /= norm_factor

        # Get the patch from im1 centered at the point
        x1, y1 = int(pt1[0]), int(pt1[1])
        patch1 = im1[max(0, y1 - half_window):y1 + half_window + 1, max(0, x1 - half_window):x1 + half_window + 1]
        # patch1 = padded_im1[y1:y1 + window_size, x1:x1 + window_size]

        best_score = float('inf')
        # min_distance = float('inf')
        best_pt2 = None
        candidate_pts = []

        # Scan along the epipolar line within image bounds
        for x2 in range(half_window, W - half_window):
            y2 = int(-(a * x2 + c) / b)
            if y2 < half_window or y2 >= H - half_window:
                continue

            # Get the candidate patch from im2
            patch2 = im2[y2 - half_window:y2 + half_window + 1, x2 - half_window:x2 + half_window + 1]

            if patch1.shape == patch2.shape:  # Ensure valid patches
                # Compute similarity (e.g., SSD)
                score = np.sum((patch1 - patch2)**2)

                if score < best_score:
                    best_score = score
                    best_pt2 = (x2, y2)
            # if half_window <= y2 < H - half_window:
            #     candidate_pts.append((x2, y2))

    # Calculate similarity (or difference) between patch in im1 and candidate patches in im2


    # Select the best match for each point in pts1

        if best_pt2:
            pts2.append(best_pt2)

    # Replace pass with output
    # pass
    # print("Returned pts2:", pts2)

    return np.array(pts2)


"""
Q2.3 Essential Matrix
   [I] F -- the fundamental matrix (3x3 matrix)
       K1 -- camera matrix 1 (3x3 matrix)
       K2 -- camera matrix 2 (3x3 matrix)
   [O] E -- the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # Replace pass with your implementation
    E = K2.T @ F @ K1
    # pass
    return E


"""
Q2.4 Triangulation
   [I] P1 -- camera projection matrix 1 (3x4 matrix)
       pts1 -- points in image 1 (Nx2 matrix)
       P2 -- camera projection matrix 2 (3x4 matrix)
       pts2 -- points in image 2 (Nx2 matrix)
   [O] pts3d -- 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass with your implementation
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))

    for i in range(num_points):
        # Construct the matrix A for each point pair
        A = np.array([
            pts1[i, 0] * P1[2, :] - P1[0, :],
            pts1[i, 1] * P1[2, :] - P1[1, :],
            pts2[i, 0] * P2[2, :] - P2[0, :],
            pts2[i, 1] * P2[2, :] - P2[1, :]
        ])

        # Solve for the 3D point using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]  # Normalize to make it homogeneous

        pts3d[i] = X[:3]

    return pts3d
    # pass


"""
Q3.1 Image Rectification
   [I] K1 K2 -- camera matrices (3x3 matrix)
       R1 R2 -- rotation matrices (3x3 matrix)
       t1 t2 -- translation vectors (3x1 matrix)
   [O] M1 M2 -- rectification matrices (3x3 matrix)
       K1p K2p -- rectified camera matrices (3x3 matrix)
       R1p R2p -- rectified rotation matrices (3x3 matrix)
       t1p t2p -- rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # Compute the optical centers of each camera
    c1 = -np.linalg.inv(R1) @ t1
    c2 = -np.linalg.inv(R2) @ t2

    # Compute T = (c1 - c2) / norm(c1 - c2)
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)

    r1 = r1.reshape(3)

    # Compute the rotation matrix R_tilde and set R1p and R2p equal to R_tilde
    r2 = np.cross(R1[2, :].reshape(3), r1) #.T
    r2 /= np.linalg.norm(r2)

    r3 = np.cross(r2, r1)
    r3 /= np.linalg.norm(r3)

    eR = np.vstack((r1, r2, r3)).T
    R1p, R2p = eR, eR

    # Compute the new intrinsic parameters as K1p = K2p = K2
    K1p, K2p = K2, K2

    # Compute the new translation vectors t1p and t2p
    t1p = -eR @ c1
    t2p = -eR @ c2

    # Compute the rectification matrices M1 and M2
    M1 = K1p @ R1p @ np.linalg.inv(K1 @ R1)
    M2 = K2p @ R2p @ np.linalg.inv(K2 @ R2)

    # replace pass with output
    # pass
    
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2 Disparity Map
   [I] im1 -- image 1 (H1xW1 matrix)
       im2 -- image 2 (H2xW2 matrix)
       max_disp -- scalar maximum disparity value
       win_size -- scalar window size value
   [O] dispM -- disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    
    half_win = (win_size - 1) // 2
    h, w = im1.shape

    kernel = np.ones((win_size, win_size), dtype=np.float32)

    dispM = np.empty(im1.shape, dtype=np.uint16)
    # # for ...
    # # dist = scipy.signal.convolve2d(diff, kernel, mode="same", boundary="symm")
    for d in range(max_disp + 1):
        # Shift im2 by disparity d
        shifted_im2 = np.roll(im2, -d, axis=1)
        
        # Calculate the SSD for each window
        ssd = (im1 - shifted_im2) ** 2
        ssd_sum = convolve2d(ssd, kernel, mode='same', boundary='symm')
        
        # Update disparity map with the smallest SSD for each pixel
        if d == 0:
            min_ssd = ssd_sum
            dispM[:] = d
        else:
            mask = ssd_sum < min_ssd
            dispM[mask] = d
            min_ssd[mask] = ssd_sum[mask]


    # replace pass by your implementation
    # pass

    return dispM


"""
Q3.2.3 Depth Map
   [I] dispM -- disparity map (H1xW1 matrix)
       K1 K2 -- camera matrices (3x3 matrix)
       R1 R2 -- rotation matrices (3x3 matrix)
       t1 t2 -- translation vectors (3x1 matrix)
   [O] depthM -- depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    # Calculate focal length f from K1 (first element in the camera matrix K1)
    f = K1[0, 0]
    
    # Calculate the baseline distance b (distance between optical centers)
    b = np.linalg.norm(t1 - t2)
    
    # Initialize the depth map with the same dimensions as disparity map
    depthM = np.zeros_like(dispM)
    
    # Calculate depth using the formula depthM(y, x) = bf / dispM(y, x)
    # Avoid division by zero by setting depthM to zero where dispM is zero
    non_zero_disp = dispM > 0
    depthM[non_zero_disp] = (b * f) / dispM[non_zero_disp]

    # pass    
    return depthM


"""
Q4.1 Camera Matrix Estimation
   [I] x -- 2D points (Nx2 matrix)
       X -- 3D points (Nx3 matrix)
   [O] P -- camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    # Number of points
    N = x.shape[0]
    # A = np.zeros((2 * N, 12))
    
    # Initialize the matrix A for DLT
    A = []
    for i in range(N):
        X_i = np.hstack((X[i], 1)) #((X[:, i], 1)) # Convert 3D point to homogeneous coordinates
        x_i, y_i = x[i] #x[0, i], x[1, i] # #x[:2, i]  # 2D point coordinates
        # Form the DLT equation
        A.append(np.hstack((-X_i, np.zeros(4), x_i * X_i)))
        A.append(np.hstack((np.zeros(4), -X_i, y_i * X_i)))
    
    # Convert A to numpy array
    A = np.array(A)
    
    # Solve Ap = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    print("Shape of A:", A.shape)
    P = Vt[-1].reshape(3, 4)  # Last row of V transposed is the solution

    # pass
    return P


"""
Q4.2 Camera Parameter Estimation
   [I] P -- camera matrix (3x4 matrix)
   [O] K -- camera intrinsics (3x3 matrix)
       R -- camera extrinsics rotation (3x3 matrix)
       t -- camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # Compute SVD on P to recover camera center class c
    #   Note: the output for c is in homogeneous coordinate system;
    #           convert to heterogeneous coordinate system.
    U, S, Vt = np.linalg.svd(P)
    c_homogeneous = Vt[-1]
    c = c_homogeneous[:3] / c_homogeneous[3] 

    # Extract M matrix (KR) from P matrix
    M = P[:, :3]

    # Run RQ decomposition on M
    K, R = scipy.linalg.rq(M)

    # The RQ decomposition solution is not unique, so we need to make the
    #   diagonal of K and the determinant of R positive.
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1
    if np.linalg.det(R) < 0:
        R = -R

    K /= K[2, 2]

    # Compute the translation t = -Rc
    t = -R @ c

    # replace pass with your output
    # pass

    return K, R, t

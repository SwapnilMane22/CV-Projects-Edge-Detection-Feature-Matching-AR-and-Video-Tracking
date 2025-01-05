import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2

"""
Part 1 (Q2): Sparse Reconstruction
"""


def main():

    # 1. Load the two temple images and the points from data/some_corresp.npz
    im1 = cv2.imread("../data/im1.png")
    im2 = cv2.imread("../data/im2.png")
    corresp = np.load("../data/some_corresp.npz")

    # OpenCV uses BGR, while matplotlib uses RGB
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    
    pts1 = corresp['pts1']  # Nx2 matrix for image 1 points
    pts2 = corresp['pts2']  # Nx2 matrix for image 2 points

    # 2. Run eight_point to compute F
    F = sub.eight_point(pts1, pts2)

    print(F)

    # Plot the fundamental matrix as a heatmap
    plt.imshow(F, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap of Fundamental Matrix F")
    plt.show()


    # This is used for visualization and debugging
    # helper.displayEpipolarF(img1, img2, F)
    hlp.displayEpipolarF(im1, im2, F)

    # 3. Load points in image 1 from data/temple_coords.npz
    pts1 = np.load("../data/temple_coords.npz")["pts1"]

    # 4. Run epipolar_correspondences to get points in image 2
    pts2 = np.zeros_like(pts1)
    for i, pt in enumerate(pts1):
        x, y = pt
        print(pt)
        [(x2, y2)] = sub.epipolar_correspondences(im1, im2, F, pt) #x, y
        pts2[i] = [x2, y2]

    # This is used for visualization and debugging
    # helper.epipolarMatchGUI(im1, im2, F)
    hlp.epipolarMatchGUI(im1, im2, F)

    # 5. Load intrinsics and compute the camera projection matrix P1
    intrinsics = np.load("../data/intrinsics.npz")
    K1 = intrinsics["K1"]
    K2 = intrinsics["K2"]
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = intrinsics["K1"] @ np.hstack((R1, t1))

    # 6. Compute essential matrix
    E = sub.essential_matrix(F, K1, K2)
    print("Computed Essential Matrix E:\n", E)
    
#     Computed Essential Matrix E:
#  [[-7.54935326e-03  9.88261785e-01  1.20159639e-01]
#  [ 5.11611769e-01 -3.32218190e-03 -5.74909704e+00]
#  [ 1.80214568e-02  5.82251181e+00  5.63155993e-03]]
# [[ 1.39330579e+03 -3.51724677e+01  6.78583380e+02 -1.46909045e+03]
#  [-2.70721169e+01  1.52601462e+03  2.44667318e+02  1.02600536e+01]
#  [-2.54019147e-01  2.19640245e-03  9.67196696e-01  1.69719347e-01]]

    # 7. Use camera2 to get 4 camera projection matrices P2
    M2_candidates = hlp.camera2(E)

    # 8. Run triangulate using the projection matrices
    pts3d = None
    min_error = float('inf')
    best_P2 = None
    max_positive_depth = 0
    best_points_3d = None

    # 9. Figure out the correct P2

    for i in range(M2_candidates.shape[2]):
        P2_candidate = K2 @ M2_candidates[:, :, i]

        pts3d_candidate = sub.triangulate(P1, pts1, P2_candidate, pts2)

        pts3d_in_cam1 = pts3d_candidate[:, 2] > 0  
        pts3d_in_cam2 = (P2_candidate @ np.hstack((pts3d_candidate, np.ones((pts3d_candidate.shape[0], 1)))).T)[2, :] > 0  

        if np.all(pts3d_in_cam1) and np.all(pts3d_in_cam2):
            error1 = hlp.reprojection_error(pts3d_candidate, pts1, P1)
            error2 = hlp.reprojection_error(pts3d_candidate, pts2, P2_candidate)

            total_error = error1 + error2

            if total_error < min_error:
                min_error = total_error
                best_P2 = P2_candidate
                best_pts3d = pts3d_candidate


    P2 = best_P2
    pts3d = best_pts3d
    print(P2)
    # pts3d = best_points_3d

    # 10. Compute the reprojection_error
    print("P1 reprojection error:", hlp.reprojection_error(pts3d, pts1, P1))
    print("P2 reprojection error:", hlp.reprojection_error(pts3d, pts2, P2))

# P1 reprojection error: 2.290673231037422
# P2 reprojection error: 2.242741953278952

    print("Best camera projection matrix P2:\n", best_P2)
# Best camera projection matrix P2:
#  [[ 1.39330579e+03 -3.51724677e+01  6.78583380e+02 -1.46909045e+03]
#  [-2.70721169e+01  1.52601462e+03  2.44667318e+02  1.02600536e+01]
#  [-2.54019147e-01  2.19640245e-03  9.67196696e-01  1.69719347e-01]]
    print("Triangulated 3D points:\n", pts3d)

    # 11. Scatter plot the correct 3D points
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts3d[:, 0], pts3d[:, 2], -pts3d[:, 1])
    #ax.axis("equal")
    #ax.set_aspect('equal', adjustable='box')
    #ax.set_box_aspect([1,1,1])
    ax.set_xlim(-1, 1)
    ax.set_ylim( 3, 5)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.tight_layout()
    plt.show()

    # 12. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
    R2_t2 = np.linalg.inv(K2) @ P2
    R2 = R2_t2[:, :3]
    t2 = R2_t2[:, 3, np.newaxis]
    np.savez("../data/extrinsics.npz", R1=R1, t1=t1, R2=R2, t2=t2)


if __name__ == "__main__":
    main()

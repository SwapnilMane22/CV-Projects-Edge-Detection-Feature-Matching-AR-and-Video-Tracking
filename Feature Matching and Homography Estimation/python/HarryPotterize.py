import numpy as np
import os
import cv2
import skimage.io
import skimage.color
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

DISPLAY = True

"""
Q3.9
"""
if __name__ == "__main__":

    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    resultsdir = "../results/"
    os.makedirs(resultsdir, exist_ok=True)

    # Get matches and corresponding locations
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)


    # Create set of points (x1, x2) corresponding to various matches
    # NOTE: Points maybe be in (y,x) not (x,y)
    matched_pts1 = locs1[matches[:, 0]]  # Points from cv_cover
    matched_pts2 = locs2[matches[:, 1]]  # Points from cv_desk
    x1 = matched_pts1[:, [1, 0]]  # Switching from (x, y) to (y, x)
    x2 = matched_pts2[:, [1, 0]]  # Switching from (x, y) to (y, x)

    # Find H and inliners using ransac
    H = None

    # Compute the homography matrix with RANSAC
    # H, inliers = computeH_ransac(matched_pts1, matched_pts2)
    H, inliers = computeH_ransac(x1, x2)

    # Resize hp cover to that of the cv_cover shape before transforming
    resizeShape = (cv_cover.shape[1], cv_cover.shape[0])
    hp_cover = cv2.resize(hp_cover, dsize=resizeShape)

    # Get composite image
    compositeImg = compositeH(H, hp_cover, cv_desk)
    saveTo = os.path.join(resultsdir, "compositeImg.png")
    cv2.imwrite(saveTo, compositeImg) 
    cv2.imshow("Composite Image", compositeImg)
    cv2.waitKey(0)

    if DISPLAY:
        # For cross-checking only; cv2 has its inputs swapped.
        H_ground_truth, inliners = cv2.findHomography(x2, x1, method=cv2.RANSAC)

        imgShape = (cv_desk.shape[1], cv_desk.shape[0])
        warpedImg = cv2.warpPerspective(hp_cover, np.linalg.inv(H), dsize=imgShape)
        cv2.imshow("My Output", warpedImg)
        saveTo1 = os.path.join(resultsdir, "OutputwarpedImg.png")
        cv2.imwrite(saveTo1, warpedImg)

        warpedImg = cv2.warpPerspective(hp_cover, np.linalg.inv(H_ground_truth), dsize=imgShape)
        cv2.imshow("Ground Truth (cv2)", warpedImg)
        saveTo2 = os.path.join(resultsdir, "Ground Truth warpedImg.png")
        cv2.imwrite(saveTo2, warpedImg) 
        cv2.waitKey(0)
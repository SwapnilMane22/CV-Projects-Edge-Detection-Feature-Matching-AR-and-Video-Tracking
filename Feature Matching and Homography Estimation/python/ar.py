import numpy as np
import cv2
import os

from loadSaveVid import loadVid, saveVid

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

arSourcePath = "../data/ar_source.mov"
bookMovieSourcePath = "../data/book.mov"
resultsdir = "../results"
videoPath = "../results/ar.avi"


"""
Q4.1
"""
def main():
    os.makedirs(resultsdir, exist_ok=True)

    # Preload frames from npy file if possible
    if (os.path.exists("../data/arFrames.npy") and os.path.exists("../data/bookFrames.npy")):
        arFrames = np.load("../data/arFrames.npy")
        bookFrames = np.load("../data/bookFrames.npy")
    # Otherwise load from video and save
    else:
        arFrames = loadVid(arSourcePath)
        bookFrames = loadVid(bookMovieSourcePath)
        np.save("../data/arFrames.npy", arFrames)
        np.save("../data/bookFrames.npy", bookFrames)

    # Load book cover
    bookCover = cv2.imread('../data/cv_cover.jpg')

    fps = 30
    compositeFrames = []

    # Start matching frame by frame
    nFrames = min(len(bookFrames), len(arFrames))

    # for frameNo in np.arange(startFrame, endFrame, skip):
    for frameNo in range(nFrames):
        try:
            arFrame = arFrames[frameNo]
            bookMovFrame = bookFrames[frameNo]

            # First crop the arFrame's black lines first

            # Crop the frame (at center) using aspect ratio from the cover picture
            arFrameCropped = cropFrameToCover(arFrame, bookCover)
            
            # Then with the cropped frame create composite frame
            compositeFrame = overlayFrame(bookCover, bookMovFrame, arFrameCropped)

            compositeFrames.append(compositeFrame)

        except Exception as e:
            print(f"Failure at frame {frameNo}, saving progress first.")
            print("Failure Reason:", e)


    # Save frames for post processing (can view or save with parseFrames)
    # Would need to write a script (in a new file) if you want to visualize per image frame
    np.save("../results/compositeFrames.npy", np.array(compositeFrames))

    saveVid(videoPath, compositeFrames)


"""
@brief Overlay pre-cropped arFrame on bookMovFrame using bookCover for matches
@param[in] bookCover Cover to overlay over
@param[in] bookMovFrame Frame from the book video (to overlay ON)
@param[in] arFrame Pre-cropped arFrame

@return New composite frame
"""
# NOTE: Because the book cover is repeated, can cache the descriptors and locs of that
# HINT HINT HINT
def overlayFrame(bookCover, bookMovFrame, arFrameCropped, threshold=10):

    #Compute features, descriptors and Match features
    orb = cv2.ORB_create()
    locs1, des1 = orb.detectAndCompute(bookCover, None)
    locs2, des2 = orb.detectAndCompute(bookMovFrame, None)

    # BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Create set of points (x1, x2) corresponding to various matches
    # NOTE: Points are in (y,x) not (x,y)
    src_pts = np.float32([locs1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([locs2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Find H and inliners using ransac
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Normalise H for a better fit (like OpenCV does)
    H /= H[2, 2]

    arWarped = cv2.warpPerspective(arFrameCropped, H, (bookMovFrame.shape[1], bookMovFrame.shape[0]))

    mask = np.any(arWarped != [0, 0, 0], axis=-1).astype(np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)
    bookFrameMasked = cv2.bitwise_and(bookMovFrame, bookMovFrame, mask=mask_inv)

    # NOTE: AR frame should already be resized during cropping
    # Get composite image
    compositeImg = compositeH(
        H, arFrameCropped, bookMovFrame, alreadyInverted=True)
    # compositeImg = cv2.add(bookFrameMasked, arWarped)

    prevCompositeImg = compositeImg

    return compositeImg


"""
Crop the frame (at center) using aspect ratio from the cover picture

@param[in] frame Frame of video to crop
@param[in] cover Cover picture
@param[in] Whether or not to resize image to min dimensions before cropping
           Otherwise, performs a center crop

@return Cropped frame
"""
def cropFrameToCover(frame, cover):

    blackLinesSize = 50

    frame = frame[blackLinesSize:-blackLinesSize, :]
    
    # Resize to fit min dimension before cropping
    frameResized = np.copy(frame)

    cover_height, cover_width = cover.shape[:2]
    frame_height, frame_width = frameResized.shape[:2]

    aspect_ratio_cover = cover_height / frame_height
    resizeShape = np.array(frame.shape[:-1]) * aspect_ratio_cover
    resizeShape = tuple(np.flip(resizeShape.astype(int)))
    frameResized = cv2.resize(frameResized, dsize=resizeShape)

    frame_height, frame_width = frameResized.shape[:2]

    coverShape = np.array(cover.shape[:2])
    frameShape = np.array(frameResized.shape[:2])

    # Calculate the center coordinates for cropping
    center_y = frame_height // 2
    center_x = frame_width // 2

    # Calculate the cropping box (center-crop logic)
    x_start = max(center_x - cover_width // 2, 0)
    x_end = min(center_x + cover_width // 2, frame_width)
    y_start = max(center_y - cover_height // 2, 0)
    y_end = min(center_y + cover_height // 2, frame_height)

    # Crop using indexes
    # frameCropped = frameResized
    frameCropped = frameResized[y_start:y_end, x_start:x_end]

    return frameCropped


if __name__ == "__main__":
    main()











# """
# Q4.1
# """
# def main():
#     os.makedirs(resultsdir, exist_ok=True)

#     # Preload frames from npy file if possible
#     if (os.path.exists("../data/arFrames.npy") and os.path.exists("../data/bookFrames.npy")):
#         arFrames = np.load("../data/arFrames.npy")
#         bookFrames = np.load("../data/bookFrames.npy")
#     # Otherwise load from video and save
#     else:
#         arFrames = loadVid(arSourcePath)
#         bookFrames = loadVid(bookMovieSourcePath)
#         np.save("../data/arFrames.npy", arFrames)
#         np.save("../data/bookFrames.npy", bookFrames)

#     # Load book cover
#     bookCover = cv2.imread('../data/cv_cover.jpg')

#     fps = 30
#     compositeFrames = []
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(bookCover, None)

#     # Start matching frame by frame
#     nFrames = min(len(bookFrames), len(arFrames))

#     # for frameNo in np.arange(startFrame, endFrame, skip):
#     for frameNo in range(nFrames):
#         try:
#             arFrame = arFrames[frameNo]
#             bookMovFrame = bookFrames[frameNo]

#             # First crop the arFrame's black lines first

#             # Crop the frame (at center) using aspect ratio from the cover picture
#             arFrameCropped = cropFrameToCover(arFrame, bookCover)
            
#             kp2, des2 = orb.detectAndCompute(bookMovFrame, None)

#             bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#             matches = bf.match(des1, des2)
#             matches = sorted(matches, key=lambda x: x.distance)

#             src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#             dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#             H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#             # arFrameCropped = cropFrameToCover(arFrame, bookCover)

#             arWarped = cv2.warpPerspective(arFrameCropped, H, (bookMovFrame.shape[1], bookMovFrame.shape[0]))

#             mask = np.any(arWarped != [0, 0, 0], axis=-1).astype(np.uint8) * 255
#             mask_inv = cv2.bitwise_not(mask)
#             bookFrameMasked = cv2.bitwise_and(bookMovFrame, bookMovFrame, mask=mask_inv)
#             compositeFrame = cv2.add(bookFrameMasked, arWarped)

#             # Then with the cropped frame create composite frame
#             # compositeFrame = overlayFrame(bookCover, bookMovFrame, arFrameCropped)

#             compositeFrames.append(compositeFrame)

#         except Exception as e:
#             print(f"Failure at frame {frameNo}, saving progress first.")
#             print("Failure Reason:", e)


#     # Save frames for post processing (can view or save with parseFrames)
#     # Would need to write a script (in a new file) if you want to visualize per image frame
#     np.save("../results/compositeFrames.npy", np.array(compositeFrames))

#     saveVid(videoPath, compositeFrames)


# """
# @brief Overlay pre-cropped arFrame on bookMovFrame using bookCover for matches
# @param[in] bookCover Cover to overlay over
# @param[in] bookMovFrame Frame from the book video (to overlay ON)
# @param[in] arFrame Pre-cropped arFrame

# @return New composite frame
# """
# # NOTE: Because the book cover is repeated, can cache the descriptors and locs of that
# # HINT HINT HINT
# def overlayFrame(bookCover, bookMovFrame, arFrameCropped, threshold=10):

#     #Compute features, descriptors and Match features

#     # Create set of points (x1, x2) corresponding to various matches
#     # NOTE: Points are in (y,x) not (x,y)

#     # Find H and inliners using ransac

#     # Normalise H for a better fit (like OpenCV does)
#     H /= H[2, 2]

#     # NOTE: AR frame should already be resized during cropping
#     # Get composite image
#     compositeImg = compositeH(
#         H, arFrameCropped, bookMovFrame, alreadyInverted=True)

#     prevCompositeImg = compositeImg

#     return compositeImg


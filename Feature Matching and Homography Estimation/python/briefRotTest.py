import os
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matchPics import matchPics
from scipy.ndimage import rotate
from helper import plotMatches

resultsdir = "../results/rotTest"

"""
Q3.5
"""
if __name__ == "__main__":
	os.makedirs(resultsdir, exist_ok=True)

	#Read the image and convert to grayscale, if necessary
	originalImg = cv2.imread("../data/cv_cover.jpg")
	rotImg = originalImg.copy()

	# Histogram count for matches
	nMatches = []
	angles = [ (i+1)*10 for i in range(36) ]

	# for i in range(36):
	for i, angle in enumerate(angles):
		#Rotate Image
		rotImg = rotate(originalImg, angle, reshape=False)

		#Compute features, descriptors and Match features
		matches, locs1, locs2 = matchPics(originalImg, rotImg)

		#Update histogram
		# nMatches.append(0) # CHANGE
		nMatches.append(len(matches))

		# Save all results
		saveTo = os.path.join(resultsdir, f"rot{(i+1)*10}.png")
		plotMatches(originalImg, rotImg, matches, locs1, locs2, saveTo=saveTo, showImg=True)


	#Display histogram
	plt.tight_layout()
	plt.bar(x=angles, height=nMatches, width=5)
	plt.show()

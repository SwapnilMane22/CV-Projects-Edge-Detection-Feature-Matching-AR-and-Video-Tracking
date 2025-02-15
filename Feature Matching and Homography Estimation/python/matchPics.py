import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


"""
Q3.4
"""
def matchPics(I1, I2, ratio=0.67):
	#I1, I2 : Images to match
	sigma = 0.15


	#Convert Images to GrayScale
	I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)


	#Detect Features in Both Images
	locs1 = corner_detection(I1_gray, sigma)
	locs2 = corner_detection(I2_gray, sigma)


	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1_gray, locs1)
	desc2, locs2 = computeBrief(I2_gray, locs2)


	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio)


	return matches, locs1, locs2

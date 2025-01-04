import numpy as np
import cv2
import random

"""
Q3.6
Compute the homography between two sets of points

@param[in] x1 Set of points x1 in hetereogeneous coords
@param[in] x2 Set of points x2 in hetereogeneous coords

@return H2to1 Closest 3x3 H matrix (least squares)
"""
def computeH(x1, x2):

	# Create A_i for each point correspondence
    # And stack together to form A
	assert x1.shape == x2.shape, "Input point sets must have the same shape"
	num_points = x1.shape[0]

	A = []

	for i in range(num_points):
	    x_1, y_1 = x1[i] #x1[i, 0], x1[i, 1] #
	    x_2, y_2 = x2[i] #x2[i, 0], x2[i, 1] #
	    
	    A.append([-x_2, -y_2, -1, 0, 0, 0, x_1 * x_2, x_1 * y_2, x_1])
	    A.append([0, 0, 0, -x_2, -y_2, -1, y_1 * x_2, y_1 * y_2, y_1])

	A = np.array(A)


	# Perform EIG or SVD of A
	U, S, Vh = np.linalg.svd(A)
	h = Vh[-1].reshape(3, 3) #Vh[-1, :]


	# Reshape h to 3x3
	H2to1 = h.reshape((3, 3))


	return H2to1


"""
Q3.7
Normalise the coordinates to reduce noise before computing H

@param[in] x1 Set of points x1 in hetereogeneous coords
@param[in] x2 Set of points x2 in hetereogeneous coords

@return H2to1 Closest 3x3 H matrix (least squares)
"""
def computeH_norm(x1, x2):

	#Compute the centroid of the points


	#Shift the origin of the points to the centroid


	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	# Basically normalize points to range between [0...1] (abs value)


	x1norm = None	# CHANGE
	x2norm = None	# CHANGE

	# NOTE: order of translation then scaling affects the similarity matrix construction:
    #   1/norm  0       -meanx/norm
    #   0       1/norm  -meany/norm
    #   0       0       1
	#
	#
	# Similarity transform 1
    # Scaling and translation
    # But using precomputed inverse:
    #   norm    0       meanx
    #      0    norm    meany
    #      0    0       1

	T1_inv = None	# CHANGE


	# Similarity transform 2
    # Translation THEN scaling
    #   1/norm  0       -meanx/norm
    #   0       1/norm  -meany/norm
    #   0       0       1

	T2 = None # CHANGE
    
	def normalize_points(points):
		"""
		Normalizes the input points by translating them so the mean is at the origin,
		and scaling them so that the average distance from the origin is sqrt(2).

		@param points: N x 2 array of (x, y) coordinates to normalize

		@return: normalized points, and the transformation matrix
		"""
		# Compute the centroid
		centroid = np.mean(points, axis=0)

		# Shift the points so that the centroid is at the origin
		shifted_points = points - centroid

		# Compute the average distance of the points from the origin
		distances = np.sqrt(np.sum(shifted_points**2, axis=1))
		avg_distance = np.mean(distances)

		# Compute the scaling factor
		scale = np.sqrt(2) / avg_distance

		# Create the similarity transformation matrix T
		T = np.array([
		    [scale, 0, -scale * centroid[0]],
		    [0, scale, -scale * centroid[1]],
		    [0, 0, 1]
		])

		# Normalize the points
		normalized_points = np.dot(T, np.vstack((points.T, np.ones((1, points.shape[0])))))
		normalized_points = normalized_points[:2, :].T  # Drop the homogeneous coordinate

		return normalized_points, T
    
	# Normalize x1 and x2
	x1norm, T1 = normalize_points(x1)
	x2norm, T2 = normalize_points(x2)

	T1_inv = np.linalg.inv(T1)

	#Compute homography
	H2to1 = computeH(x1norm, x2norm)

	#Denormalization
	H2to1 = T1_inv @ H2to1 @ T2

	return H2to1


"""
Q3.8
Run RANSAC on set of matched points x1, x2.
Reduces effect of outliers by finding inliers.
Returns best fitting homography H and best inlier set.

@param[in] x1 Set of points x1 in hetereogeneous coords
@param[in] x2 Set of points x2 in hetereogeneous coords
@param[in] threshold
    # TODO: Find out what the threshold is
    # Note that threshold is squared sum of difference
    # to avoid extra sqrt computation, so threshold
    # will be number of pixels away, SQUARED
    threshold = 10  # ~3 pixels away

@return bestH2to1
@return bestInlier Vector of length N with a 1 at matches, 0 elsewhere
"""
def computeH_ransac(x1, x2, nSamples=None, threshold=10):
	num_points = x1.shape[0]
	best_inliers = None
	max_inliers = 0
	bestH2to1 = None
	nSamples = 10000

	for _ in range(nSamples):
	    # Randomly select 4 point correspondences
	    idx = random.sample(range(num_points), 4)
	    x1_sample = x1[idx, :]
	    x2_sample = x2[idx, :]

	    # Compute homography using the 4-point sample
	    H2to1 = computeH_norm(x1_sample, x2_sample)

	    # Transform all points x2 using the computed homography
	    x2_homogeneous = np.hstack([x2, np.ones((num_points, 1))])  # Make x2 homogeneous
	    x2_transformed = (H2to1 @ x2_homogeneous.T).T  # Apply homography
	    x2_transformed = x2_transformed[:, :2] / x2_transformed[:, [2]]  # Normalize the points

	    # Compute the squared Euclidean distance between x1 and transformed x2
	    distances = np.sum((x1 - x2_transformed)**2, axis=1)

	    # Determine inliers based on the threshold
	    inliers = distances < threshold

	    # Count the number of inliers
	    num_inliers = np.sum(inliers)

	    # Update best homography if more inliers are found
	    if num_inliers > max_inliers:
	        max_inliers = num_inliers
	        best_inliers = inliers
	        bestH2to1 = H2to1

	return bestH2to1, best_inliers #inliers


"""
Q3.9
Create a composite image after warping the template image on top
of the image using the homography

Note that the homography we compute is from the image to the template;
x_template = H2to1*x_photo
"""
def compositeH(H2to1, template, img, alreadyInverted=False):

	# For warping the template to the image, we need to invert it.
	# Might already be inverted... (video problem)
	# H1to2 = None # CHANGE
	if not alreadyInverted:
		H1to2 = np.linalg.inv(H2to1)
	else:
		H1to2 = H2to1

	# Create mask of same size as template
	mask = None # CHANGE
	mask = np.ones(template.shape, dtype=np.uint8) * 255

	# Warp mask by appropriate homography
	# mask = None # CHANGE
	mask = cv2.warpPerspective(mask, H1to2, (img.shape[1], img.shape[0]))

	# Warp template by appropriate homography
	templateWarped = None # CHANGE
	templateWarped = cv2.warpPerspective(template, H1to2, (img.shape[1], img.shape[0]))

	mask_inv = cv2.bitwise_not(mask)

	img_masked = cv2.bitwise_and(img, img, mask=mask_inv[:, :, 0])

	# Use mask to combine the warped template and the image
	# composite_img = np.add(np.multiply(img, mask), templateWarped)
	composite_img = cv2.add(img_masked, templateWarped)

	return composite_img

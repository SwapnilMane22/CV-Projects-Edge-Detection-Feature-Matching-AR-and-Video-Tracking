import numpy as np

def myImageFilter(img0, h):
    # YOUR CODE HERE
	img_height, img_width = img0.shape
	filter_height, filter_width = h.shape

	pad_height = filter_height // 2
	pad_width = filter_width // 2

	padded_img = np.pad(img0, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

	img1 = np.zeros_like(img0)

	patches = np.zeros((img_height, img_width, filter_height, filter_width))

	for i in range(filter_height):
	    for j in range(filter_width):
	        patches[:, :, i, j] = padded_img[i:i+img_height, j:j+img_width]

	img1 = np.sum(patches * h, axis=(2, 3))

	return img1
import numpy as np
import cv2

def myHoughLineSegments(lineRho, lineTheta, Im, img_lines):
	minLineLength=10
	maxLineGap=3

	lines = []

	height, width = Im.shape

	for i in range(len(lineRho)):
	    rho = lineRho[i]
	    theta = lineTheta[i]

	    a = np.cos(theta)
	    b = np.sin(theta)

	    x0 = a * rho
	    y0 = b * rho

	    x1 = int(x0 + 1000 * (-b)) 
	    y1 = int(y0 + 1000 * (a))
	    x2 = int(x0 - 1000 * (-b))
	    y2 = int(y0 - 1000 * (a))

	    x1 = np.clip(x1, 0, width)
	    y1 = np.clip(y1, 0, height)
	    x2 = np.clip(x2, 0, width)
	    y2 = np.clip(y2, 0, height)

	    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

	    if line_length >= minLineLength:
	        lines.append(((x1, y1), (x2, y2)))

	        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)

	return lines, img_lines
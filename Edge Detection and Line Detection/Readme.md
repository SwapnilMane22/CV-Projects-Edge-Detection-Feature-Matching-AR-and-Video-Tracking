# Edge Detection and Line Detection

## Overview
Edge Detection and Line Detection is a Python-based image processing tool designed for detecting lines in images using the Hough Transform. This project integrates convolution-based image filtering, edge detection, and parameterized line detection techniques, enabling precise line fitting to image edges. The application is optimized for performance, ensuring efficient computation even with high-resolution images.

## Features
- **Image Convolution**: Implements a custom convolution filter for image smoothing and preprocessing.
- **Edge Detection**: Detects edges using Gaussian smoothing and Sobel operators with non-maximum suppression for thinning edges.
- **Hough Transform**: Applies parameterized line detection through a custom Hough Transform algorithm.
- **Parameter Tuning**: Allows dynamic adjustment of Hough Transform parameters such as rho and theta resolutions.
- **Visualization Tools**: Generates and visualizes Hough space sinusoidal curves and detected lines for analysis.
- **Extra Credit Features**: Supports Hough line segmentation and custom test image evaluations.

## Installation
1. Clone the repository:
   ```bash
   https://github.com/SwapnilMane22/CV-Projects-Edge-Detection-Feature-Matching-AR-and-Video-Tracking.git
   ```
2. Navigate to the directory:
   ```bash
   cd 'Edge Detection and Line Detection'
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Organize data in the following folder structure:
   ```
   project_root/
   ├── python/
   │   ├── houghScript.py
   │   ├── myImageFilter.py
   │   ├── myEdgeFilter.py
   │   ├── myHoughTransform.py
   │   ├── myHoughLines.py
   │   ├── ec/
   │   │   ├── myHoughLineSegments.py
   │   │   ├── ec.py
   │   │   ├── images/
   │   │   ├── results/
   ├── data/
   ├── results/
   ```
2. Run the main script for line detection:
   ```bash
   python python/houghScript.py
   ```
3. Outputs are saved in the results/ directory.

## Parameters
- **Edge Detection**:
  - Sigma: Controls Gaussian smoothing level.
- **Hough Transform**:
  - Rho Resolution (rhoRes): Distance resolution in pixels.
  - Theta Resolution (thetaRes): Angular resolution in radians.
  - Threshold: Minimum votes to detect a line.

## Examples
### Convolution and Edge Detection
```python
from myImageFilter import myImageFilter
from myEdgeFilter import myEdgeFilter

img = cv2.imread('data/sample.jpg', 0)
smoothed = myImageFilter(img, h)
edges = myEdgeFilter(smoothed, sigma=1.5)
```

### Hough Transform
```python
from myHoughTransform import myHoughTransform

hough_space, rho, theta = myHoughTransform(edges, 1, np.pi/180)
```

## Results
The output images include edge-detected versions and Hough space visualizations, demonstrating line intersections in parameter space. Detected lines are overlaid on original images for comparison.

## Extra Credit
- **Line Segmentation**: Implements segmentation of detected lines.
- **Custom Images**: Allows users to test the algorithm on custom images.

## Notes
- Ensure Python scripts are vectorized for optimal performance.
- File paths must be relative to avoid dependency on system-specific directories.
- Extra credit features are stored in the `ec/` directory.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- OpenCV

## Contributions
Contributions are welcome! Submit pull requests or issues to enhance features or fix bugs.

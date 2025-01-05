# Computer Vision: Edge Detection, Feature Matching, Augmented Reality, and Video Tracking

This repository contains various implementations of computer vision and image processing techniques. The projects span a wide range of tasks, including edge detection, feature matching, 3D reconstruction, and video tracking. The repository is intended to showcase methods for solving real-world problems in the fields of machine vision, augmented reality, and video analysis.

## Projects

### 1. **Edge Detection & Line Detection**
   - **Description**: This project implements edge and line detection algorithms, including Gaussian filters, Sobel operators, and the Hough transform. The goal is to detect and highlight edges and lines in images to aid in further image analysis tasks.
   - **Key Components**: 
     - Gaussian filters and Sobel operators for edge detection.
     - Non-maximum suppression for refined edge detection.
     - Hough transform for detecting lines.

### 2. **Feature Matching & Homography Estimation**
   - **Description**: This project focuses on feature detection, matching, and homography estimation using methods such as FAST corner detection and BRIEF descriptors. It includes algorithms for matching features across images and estimating planar homographies to align images in an augmented reality setup.
   - **Key Components**:
     - Feature detection using FAST and BRIEF.
     - Homography computation and planar transformations.
     - Augmented reality integration through 3D model projections.

### 3. **3D Reconstruction & Pose Estimation**
   - **Description**: This project covers techniques for 3D reconstruction, including fundamental matrix estimation, triangulation, and camera pose estimation. The goal is to generate accurate 3D models from 2D images and estimate the positions and orientations of cameras involved in the reconstruction process.
   - **Key Components**:
     - Fundamental matrix estimation.
     - Triangulation and epipolar geometry.
     - Depth map generation and camera pose estimation.

### 4. **Video Tracking**
   - **Description**: This project implements several video tracking algorithms, including Lucas-Kanade and Inverse Compositional Alignment with affine transformations. The focus is on tracking moving objects across frames, accounting for changes in object displacement, visibility, and image conditions.
   - **Key Components**:
     - Lucas-Kanade tracking with translation and affine transformations.
     - Inverse compositional alignment for robust tracking.
     - Performance evaluation under dynamic conditions.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: OpenCV, NumPy, Matplotlib (for visualization), etc.

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### Running the Projects
1. Clone this repository:
   ```bash
   https://github.com/SwapnilMane22/CV-Projects-Edge-Detection-Feature-Matching-AR-and-Video-Tracking.git
   ```
2. Navigate to the respective project directory.
3. Run the relevant Python script

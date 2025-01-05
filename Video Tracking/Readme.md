# Video Tracking

## Overview
This project explores the implementation of video tracking algorithms, focusing on object tracking in dynamic scenes. The primary objective is to develop a tracker capable of following an object throughout a video sequence by estimating transformations between frames.

The implemented algorithms include:
1. **Lucas-Kanade Tracker** - Translation-based and affine transformation tracking.
2. **Matthews-Baker Method** - An optimized version of the Lucas-Kanade tracker for computational efficiency.
3. **Extended Features (Optional)** - Robust tracking with illumination invariance and multi-scale tracking using image pyramids.

## Background
### Image Transformations
Image transformations, or warps, map pixel coordinates and values from one location to another. Examples include translation, rotation, and scaling. This project focuses on affine transformations, parameterized by six parameters. Affine transforms can describe linear transformations, making them suitable for object tracking.

### Lucas-Kanade Method
The Lucas-Kanade algorithm aligns a sequence of images with a template by minimizing pixel-wise differences between the warped image and the template. It uses gradient-based optimization to iteratively estimate warp parameters. This approach works for small displacements and provides robust performance for object tracking.

### Matthews-Baker Method
The Matthews-Baker method improves upon Lucas-Kanade by precomputing Hessians and Jacobians, making the tracking process faster. It utilizes an inverse compositional alignment framework, leveraging invertible affine transformations.

## Project Structure
### Directory Layout
```
<Bnumber>.zip
    ├── <Bnumber>.pdf (Report and insights)
    ├── python/
    │   ├── LucasKanade.py
    │   ├── LucasKanadeAffine.py
    │   ├── InverseCompositionAffine.py
    │   ├── test_lk.py
    │   ├── test_lk_affine.py
    │   ├── test_ic_affine.py
    │   ├── utils.py
    │   ├── LucasKanadeRobust.py (optional)
    │   ├── LucasKanadePyramid.py (optional)
```

## Key Files
- **LucasKanade.py** - Implements the basic Lucas-Kanade tracker with translation-only alignment.
- **LucasKanadeAffine.py** - Extends the tracker to handle affine transformations.
- **InverseCompositionAffine.py** - Implements the Matthews-Baker method for computational efficiency.
- **test_lk.py** - Tests for translation-only tracking.
- **test_lk_affine.py** - Tests for affine tracking.
- **test_ic_affine.py** - Tests for Matthews-Baker alignment.
- **utils.py** - Contains helper functions for image processing and visualization.
- **LucasKanadeRobust.py (optional)** - Implements robust tracking with illumination invariance.
- **LucasKanadePyramid.py (optional)** - Implements multi-scale tracking using pyramids.

## Features
1. Object tracking using Lucas-Kanade and Matthews-Baker methods.
2. Support for affine transformations, including scaling and rotation.
3. Robust tracking with illumination invariance (optional).
4. Multi-scale tracking using pyramids for handling large displacements (optional).

## Usage Instructions
1. Ensure Python 3.x and required libraries (NumPy, OpenCV) are installed.
2. Organize video data and scripts as per the provided directory structure.
3. Execute test scripts for validating each algorithm:
   ```
   python test_lk.py
   python test_lk_affine.py
   python test_ic_affine.py
   ```
4. Modify parameters in the scripts as needed for specific video sequences.

## Tips
- Start with translation-based tracking before advancing to affine transformations.
- Verify each implementation step to avoid debugging complex issues later.
- Use optional modules for improved performance in challenging scenarios.

## References
1. Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision.
2. Matthews, I., & Baker, S. (2004). Active Appearance Models Revisited.

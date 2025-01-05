# 3D Reconstruction and Pose Estimation

## Description
This project explores techniques for 3D reconstruction, including fundamental matrix estimation, triangulation, and camera pose estimation. The objective is to generate accurate 3D models from 2D images and estimate the positions and orientations of the cameras involved in the reconstruction process.

## Key Components
- **Fundamental Matrix Estimation:** Computes relationships between pairs of points in two images.
- **Triangulation and Epipolar Geometry:** Recovers 3D point positions from 2D correspondences.
- **Depth Map Generation and Camera Pose Estimation:** Determines camera extrinsics and constructs depth maps for 3D visualization.

## Features
- Sparse reconstruction with epipolar geometry.
- Dense reconstruction through image rectification and depth map generation.
- Visualization of epipolar lines and point correspondences.
- Projection of 3D CAD models onto image data.

## File Structure
```
<Bnumber>.zip
├── <Bnumber>.pdf
├── python
│   ├── submission.py
│   ├── helper.py
│   ├── project_cad.py
│   ├── test_depth.py
│   ├── test_params.py
│   ├── test_pose.py
│   ├── test_rectify.py
│   ├── test_temple_coords.py
├── data
│   ├── im1.png
│   ├── im2.png
│   ├── intrinsics.npz
│   ├── some_corresp.npz
│   ├── temple_coords.npz
│   ├── pnp.npz
│   ├── extrinsics.npz
│   ├── rectify.npz
```

## Instructions
1. Ensure Python 3.x and required libraries (e.g., NumPy, OpenCV, Matplotlib) are installed.
2. Use the provided data files to load camera intrinsics and point correspondences.
3. Implement the following components in `submission.py`:
   - Fundamental matrix estimation.
   - Epipolar correspondence search.
   - Triangulation of 3D points.
   - Camera pose estimation and rectification.
4. Execute test scripts to validate the implementation.
5. Generate visualizations of epipolar lines and 3D reconstructions.
6. Optionally, compute dense depth maps and overlay CAD models for enhanced visualizations.

## Dense Reconstruction
### 3.1 Image Rectification
Complete the following function in submission.py that computes rectification matrices:
```
M1,M2,K1p,K2p,R1p,R2p,t1p,t2p = rectify_pair(K1,K2,R1,R2,t1,t2)
```
Steps:
1. Compute the optical centers of each camera.
2. Compute new rotation matrices based on camera reference frames.
3. Update intrinsic parameters and translation vectors.
4. Compute rectification matrices.

Test using:
```
python/test_rectify.py
```
Include results of horizontal epipolar lines and corresponding points in the write-up.

### 3.2 Dense Window Matching for Disparity
Implement the following function in submission.py to create a disparity map:
```
dispM = get_disparity(im1,im2,max_disp,win_size)
```
Steps:
1. Use rectified images for 1D search.
2. Compute disparity based on window size and distance metric.
3. Test the disparity map using:
```
python/test_depth.py
```
Include disparity map images in the write-up.

### 3.3 Depth Map
Implement the following function in submission.py to compute a depth map:
```
depthM = get_depth(dispM,K1,K2,R1,R2,t1,t2)
```
Steps:
1. Use baseline and focal length to compute depth.
2. Avoid division by zero for invalid disparities.
3. Test depth map using:
```
python/test_depth.py
```
Include depth map images in the write-up.

## Pose Estimation
### Camera Extrinsics
Compute extrinsic parameters based on matched points between the CAD model and the images.
Steps:
1. Load 3D coordinates of CAD model points.
2. Use `cv2.solvePnP` to find rotation and translation vectors.
3. Project CAD model points into the image plane for visualization.

### CAD Model Projection
Overlay the CAD model onto the image using:
```
cv2.projectPoints()
```
Test results using:
```
python/project_cad.py
```
Include rendered images of projected CAD models in the write-up.

## Testing and Validation
Run the provided test scripts to validate each step:
- `test_temple_coords.py` for epipolar correspondences.
- `test_pose.py` for camera pose estimation.
- `test_rectify.py` for rectification and depth map generation.
- `test_depth.py` for dense reconstruction.

## Visualization Tools
- Use `displayEpipolarF` for interactive epipolar line visualization.
- Plot 3D points and dense reconstructions for validation.

## Applications
This project demonstrates techniques applicable to fields such as robotics, autonomous navigation, and 3D modeling, enabling systems to infer depth and spatial relationships from 2D image data.

## Notes
- Avoid external code unless specified.
- Follow the submission structure strictly.
- Include visual outputs and explanations in the final documentation.

## Requirements
- Python 3.x
- NumPy
- OpenCV
- Matplotlib

## Acknowledgments
This work incorporates example datasets and inspiration from computer vision methodologies focused on 3D reconstruction and pose estimation.


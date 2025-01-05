import numpy as np

# write your implementation here

# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import submission as sub

# Load the required data
data = np.load('../data/pnp.npz', allow_pickle=True)
image = data['image']   # Load the input image
cad_model = data['cad'] # Load CAD model data (assuming it's 3D vertices)
x = data['x']           # Given 2D points in the image
X = data['X']           # Given 3D points corresponding to x
vertices = cad_model['vertices']

# Unwrap the nested structure of vertices if necessary
if isinstance(vertices, np.ndarray) and vertices.size == 1:
    vertices = vertices[0]  # Access the nested content if it's wrapped in another array
    print("Accessed vertices content:", vertices)  # Debugging: check the accessed content

# Check if the extracted vertices are still nested inside an additional array
if isinstance(vertices, np.ndarray) and vertices.size == 1 and isinstance(vertices[0], np.ndarray):
    vertices = vertices[0]  # Access the actual array of points
    print("Final vertices content shape:", vertices.shape)  # Final check

# Ensure vertices is in shape (N, 3)
if vertices.ndim != 2 or vertices.shape[1] != 3:
    vertices = vertices.reshape(-1, 3)
    print("Reshaped vertices shape:", vertices.shape)  

# Step 2: Estimate the camera matrix P, intrinsic matrix K, rotation matrix R, and translation vector t
P = sub.estimate_pose(x, X)
K, R, t = sub.estimate_params(P)

# Step 3: Project the 3D points X onto the image using the camera matrix P
X_homogeneous = np.hstack((X, np.ones((X.shape[0], 1))))  # Convert X to homogeneous coordinates
projected_points = (P @ X_homogeneous.T).T
projected_points = (projected_points[:, :2].T / projected_points[:, 2]).T  # Normalize to 2D

# Step 4: Plot given 2D points and projected 3D points on the image
plt.figure()
plt.imshow(image, cmap='gray')
plt.scatter(x[:, 0], x[:, 1], color='green', label='Given 2D points')
plt.scatter(projected_points[:, 0], projected_points[:, 1], color='red', label='Projected 3D points')
plt.legend()
plt.title("Image with Given 2D Points (Green) and Projected 3D Points (Red)")
plt.show()

# Step 5: Draw the CAD model rotated by the estimated rotation R
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# cad_model = cad_model.reshape(-1, 3) 
# vertices = vertices.reshape(-1, 3) 
rotated_cad_model = (R @ vertices.T).T  #(R @ cad_model.T).T  # Rotate the CAD model by R
# ax.scatter(rotated_cad_model[:, 0], rotated_cad_model[:, 1], rotated_cad_model[:, 2], c='k', marker='o')
ax.plot(rotated_cad_model[:, 0], rotated_cad_model[:, 1], rotated_cad_model[:, 2], 'k--')
ax.set_title("Rotated CAD Model")
plt.show()

# Step 6: Project all CAD model vertices onto the image and overlay on the image
cad_model_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1)))) #np.hstack((cad_model, np.ones((cad_model.shape[0], 1))))
projected_cad_model = (P @ cad_model_homogeneous.T).T
projected_cad_model = (projected_cad_model[:, :2].T / projected_cad_model[:, 2]).T  # Normalize to 2D

plt.figure()
plt.imshow(image, cmap='gray')
plt.scatter(x[:, 0], x[:, 1], color='green', label='Given 2D points')
plt.scatter(projected_points[:, 0], projected_points[:, 1], color='black', label='Projected 3D points')
plt.plot(projected_cad_model[:, 0], projected_cad_model[:, 1], 'r', label='Projected CAD Model', alpha=0.5)
# plt.scatter(projected_cad_model[:, 0], projected_cad_model[:, 1], color='r', label='Projected CAD Model', alpha=0.5)

plt.legend()
plt.title("Image with Projected CAD Model Overlaid")
plt.show()

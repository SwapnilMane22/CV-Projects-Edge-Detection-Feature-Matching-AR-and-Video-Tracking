import numpy as np
import matplotlib.pyplot as plt

# Define the points
points = np.array([[10, 10], [20, 20], [30, 30]])

# Create a range of theta values
theta = np.linspace(0, 2 * np.pi, 360)

# Calculate rho for each point for each theta
rho_values = []
for (x, y) in points:
    rho = x * np.cos(theta) + y * np.sin(theta)
    rho_values.append(rho)

# Plotting
plt.figure(figsize=(10, 6))
for i, rho in enumerate(rho_values):
    plt.plot(rho, theta, label=f'Point {points[i]}')

# Add labels and title
plt.title('Sinusoids in Hough Space')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\theta$ (radians)')
plt.axhline(y=np.pi/4, color='k', linestyle='--', label='Line θ = π/4')  # line at 45 degrees
plt.xlim(-50, 50)
plt.ylim(0, 2 * np.pi)
plt.legend()
plt.grid()
plt.show()

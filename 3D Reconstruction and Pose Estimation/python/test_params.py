import numpy as np
import submission as sub
import numpy.linalg as la

"""
Part 2 (Q4.2): Pose Estimation -- Estimate Intrinsic/Extrinsic Parameters
"""

# 1. Generate random camera matrix

K = np.array([[1,0,100], [0,1,100], [0,0,1]])
R, _,_ = la.svd(np.random.randn(3,3))
if la.det(R) < 0: R = -R
t = np.vstack((np.random.randn(2,1), 1))

P = K @ np.hstack((R, t))

# 2. Generate random 2D and 3D points

N = 100

X = np.random.randn(N,3)
x = P @ np.hstack((X, np.ones((N,1)))).T
x = x[:2,:].T / np.vstack((x[2,:], x[2,:])).T

# 3. Test parameter estimation with clean points

Pc = sub.estimate_pose(x, X)
Kc, Rc, tc = sub.estimate_params(Pc)

print('Intrinsic Error with clean 2D points:', la.norm((Kc/Kc[-1,-1])-(K/K[-1,-1])))
print('Rotation Error with clean 2D points:', la.norm(R-Rc))
print('Translation Error with clean 2D points:', la.norm(t-tc))

# 4. Test parameter estimation with noisy points

x = x + np.random.rand(x.shape[0], x.shape[1])
Pn = sub.estimate_pose(x, X)
Kn, Rn, tn = sub.estimate_params(Pn)

print('Intrinsic Error with noisy 2D points:', la.norm((Kn/Kn[-1,-1])-(K/K[-1,-1])))
print('Rotation Error with noisy 2D points:', la.norm(R-Rn))
print('Translation Error with noisy 2D points:', la.norm(t-tn))
# Shape of A: (200, 12)
# Intrinsic Error with clean 2D points: 2.129267966144308e-12
# Rotation Error with clean 2D points: 5.417741868257751e-13
# Translation Error with clean 2D points: 4.355797007936913
# Shape of A: (200, 12)
# Intrinsic Error with noisy 2D points: 0.7033879087401405
# Rotation Error with noisy 2D points: 0.0892079841669433
# Translation Error with noisy 2D points: 4.414566115941688
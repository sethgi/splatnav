#%%
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from plotting.plot_utils import plot_ellipse, plot_polytope
import polytope
from polytopes.decomposition import compute_polytope
from ellipsoids.intersection_utils import compute_intersection_linear_motion
from polytopes.polytopes_utils import compute_path_in_polytope
from matplotlib import pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The point of this script is to test if the ellipsoid line test and the polytope generation is working.
# We will initialize Gaussians in a C shape with the robot starting at the center and moving outward.

# Parameters
n_gaussians = 10        # per side
n_dim = 2
t = np.linspace(0, 1, n_gaussians)

# First side
quats1 = np.random.rand(n_gaussians, 4)
quats1[:, :2] = 0.      # Only rotates about z-axis
quats1 = quats1 / np.linalg.norm(quats1, axis=1)[:, None]
rots1 = Rotation.from_quat(quats1).as_matrix()
scales1 = np.random.rand(n_gaussians, n_dim) * 0.1 + 0.05

delta_x = np.array([1., 0.])
means1 = np.array([-0.5, 0.5])[None, :] + delta_x[None, :] * t[:, None]

# second side
quats2 = np.random.rand(n_gaussians, 4)
quats2[:, :2] = 0.      # Only rotates about z-axis
quats2 = quats2 / np.linalg.norm(quats2, axis=1)[:, None]
rots2 = Rotation.from_quat(quats2).as_matrix()
scales2 = np.random.rand(n_gaussians, n_dim) * 0.1 + 0.05

delta_x = np.array([0., 1.])
means2 = np.array([-0.5, -0.5])[None, :] + delta_x[None, :] * t[:, None]

# third side
quats3 = np.random.rand(n_gaussians, 4)
quats3[:, :2] = 0.      # Only rotates about z-axis
quats3 = quats3 / np.linalg.norm(quats3, axis=1)[:, None]
rots3 = Rotation.from_quat(quats3).as_matrix()
scales3 = np.random.rand(n_gaussians, n_dim) * 0.1 + 0.05

delta_x = np.array([-1., 0.])
means3 = np.array([0.5, -0.5])[None, :] + delta_x[None, :] * t[:, None]

quats = np.concatenate([quats1, quats2, quats3], axis=0)
rots = np.concatenate([rots1, rots2, rots3], axis=0)[..., :n_dim, :n_dim]
scales = np.concatenate([scales1, scales2, scales3], axis=0)
means = np.concatenate([means1, means2, means3], axis=0)

# Convert to tensors
rots = torch.tensor(rots, dtype=torch.float32, device=device)
scales = torch.tensor(scales, dtype=torch.float32, device=device)
means = torch.tensor(means, dtype=torch.float32, device=device)

#%%

delta_x = torch.tensor([0.75, 0.0], device=device)
x0 = torch.tensor([-0.25, 0.], device=device)

# robot parameters
radius = 0.25

torch.cuda.synchronize()
tnow = time.time()
output = compute_intersection_linear_motion(x0, delta_x, rots, scales, means, 
                                   R_B=None, S_B=radius, collision_type='sphere', 
                                   mode='bisection', N=10)
torch.cuda.synchronize()
print('Time to compute intersections:', time.time() - tnow)
#%% Plot ellipsoids
fig, ax = plt.subplots(1, figsize=(10, 10))

robot_plot_kwargs = {
    'facecolor': 'blue',
    'edgecolor': None,
    'alpha': 0.5
}
robot_line = x0[None, :] + torch.tensor(t[:, None], device=device) * delta_x[None, :]
Sigma_B = torch.eye(n_dim) * (radius**2)

for pt in robot_line:
    plot_ellipse(pt, Sigma_B, 1., ax, **robot_plot_kwargs)

not_intersects = output['is_not_intersect']
for (rot, scale, mean, not_intersect) in zip(rots, scales, means, not_intersects):
    
    if not_intersect:
        color = 'green'
    else:
        color = 'red'

    ellipsoid_plot_kwargs = {
    'facecolor': color,
    'edgecolor': None,
    'alpha': 0.5
    }
    # Compute covariances
    Q = rot * scale[None, :]
    Sigma = Q @ Q.T

    plot_ellipse(mean, Sigma, 1., ax, **ellipsoid_plot_kwargs)


ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
# %%

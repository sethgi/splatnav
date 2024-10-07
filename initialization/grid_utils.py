import numpy as np
import torch
import open3d as o3d

class GSplatVoxel():
    def __init__(self, gsplat, lower_bound, upper_bound, resolution, radius, device):
        self.gsplat = gsplat
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.resolution = resolution    # Can be a vector (discretization per dimension) or a scalar
        if isinstance(self.resolution, int):
            self.resolution = torch.tensor([self.resolution, self.resolution, self.resolution], device=self.device)
        
        self.radius = radius            # Robot radius
        self.device = device

        # Define max and minimum indices
        self.min_index = torch.zeros(3, dtype=int, device=self.device)
        self.max_index = torch.tensor(self.resolution, dtype=int, device=self.device) - 1
        self.cell_sizes = (upper_bound - lower_bound) / self.resolution[None, :]

        self.create_navigable_grid()

    # We employ a subdividing strategy to populate the voxel grid in order to avoid
    # having to check every point/index in the grid with all bounding boxes in the scene

    # NOTE: Might be useful to visualize this navigable grid to see if it is correct and for paper.
    def create_navigable_grid(self):

        # Create a grid
        self.navigable_grid = torch.zeros(self.resolution, dtype=bool, device=self.device)

        # ...along with its corresponding grid centers
        X, Y, Z = torch.meshgrid(
            torch.linspace(self.lower_bound[0], self.upper_bound[0], self.resolution[0]),
            torch.linspace(self.lower_bound[1], self.upper_bound[1], self.resolution[1]),
            torch.linspace(self.lower_bound[2], self.upper_bound[2], self.resolution[2])
        )
        grid_points = torch.stack([X, Y, Z], dim=-1)

        # Compute the bounding box properties, accounting for robot radius inflation
        bb_min = self.gsplat.means - torch.sqrt(torch.diagonal(self.gsplat.covs)) - self.radius
        bb_max = self.gsplat.means + torch.sqrt(torch.diagonal(self.gsplat.covs)) + self.radius
        #bb_center = self.gsplat.means

        # Optional?: Mask out ellipsoids that have bounding boxes outside of grid bounds

        # Subdivide the bounding box until it fits into the grid resolution
        lowers = []
        uppers = []
        while len(bb_min) > 0:
            # Compute the size of the bounding box
            bb_size = bb_max - bb_min

            # Check if the bounding box fits into the grid resolution
            mask = torch.all(bb_size - self.resolution[None, :] <= 0., dim=-1)      # If true, the bounding box fits into the grid resolution and we pop

            # TODO:??? Do we need to check if what we append is only one element?
            lowers.append(bb_min[mask])
            uppers.append(bb_max[mask])

            # If the bounding box does not fit within the subdivisions, divide the max dimension by 2

            # First, record the remaining bounding boxes
            bb_min = bb_min[~mask]
            bb_max = bb_max[~mask]
            bb_size = bb_size[~mask]

            # Calculate the ratio in order to know which dimension to divide by
            bb_ratio = bb_size / self.resolution[None, :]
            max_dim = torch.argmax(bb_ratio, dim=-1)
            indices_to_change = torch.stack([torch.arange(max_dim.shape[0]), max_dim], dim=-1)  # N x 2

            # Create a left and right partition (effectively doubling the size of the bounding box variables)
            bb_min_1 = bb_min.clone()
            bb_min_2 = bb_min.clone()
            bb_min_2[indices_to_change[:, 0], indices_to_change[:, 1]] += 0.5 * bb_size[indices_to_change[:, 0], indices_to_change[:, 1]]

            bb_max_1 = bb_max.clone()
            bb_max_1[indices_to_change[:, 0], indices_to_change[:, 1]] -= 0.5 * bb_size[indices_to_change[:, 0], indices_to_change[:, 1]]
            bb_max_2 = bb_max.clone()

            bb_min = torch.cat([bb_min_1, bb_min_2], dim=0)
            bb_max = torch.cat([bb_max_1, bb_max_2], dim=0)

        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)

        # Compute vertices for all subdivisions
        bb_size = bb_max - bb_min

        # The vertices are min, max, min + x, min + y, min + z, min + xy, min + xz, min + yz
        bb_min_plus_x = bb_min.clone()
        bb_min_plus_x[:, 0] += bb_size[:, 0]

        bb_min_plus_y = bb_min.clone()
        bb_min_plus_y[:, 1] += bb_size[:, 1]

        bb_min_plus_z = bb_min.clone()
        bb_min_plus_z[:, 2] += bb_size[:, 2]

        bb_min_plus_xy = bb_min.clone()
        bb_min_plus_xy[:, [0, 1]] += bb_size[:, [0, 1]]

        bb_min_plus_xz = bb_min.clone()
        bb_min_plus_xz[:, [0, 2]] += bb_size[:, [0, 2]]

        bb_min_plus_yz = bb_min.clone()
        bb_min_plus_yz[:, [1, 2]] += bb_size[:, [1, 2]]

        vertices = torch.cat([
            bb_min,
            bb_max,
            bb_min_plus_x,
            bb_min_plus_y,
            bb_min_plus_z,
            bb_min_plus_xy,
            bb_min_plus_xz,
            bb_min_plus_yz
        ], dim=0)

        # Bin the vertices into the navigable grid
        shifted_vertices = vertices - self.lower_bound          # N x 3

        vertex_index = torch.round( shifted_vertices / self.resolution[None, :] )

        # Check if the vertex or subdivision is within the grid bounds. If not, ignore.
        in_grid = ( torch.all( (self.max_index - vertex_index) >= 0. , dim=-1) ) & ( torch.all( vertex_index >= 0. , dim=-1) ) 
        vertex_index = vertex_index[in_grid]
        self.navigable_grid[vertex_index[:,0], vertex_index[:,1], vertex_index[:,2]] = True

        return
    
    def create_mesh(self):
        # Create a mesh from the navigable grid
        mesh = o3d.geometry.TriangleMesh.create_from_voxel_grid(self.navigable_grid, voxel_size=self.resolution[0], origin=self.lower_bound, color=[0.5, 0.5, 0.5])
        return mesh
  
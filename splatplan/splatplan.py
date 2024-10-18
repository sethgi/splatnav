import torch
import numpy as np
import open3d as o3d
import scipy
from scipy import sparse
import clarabel
import time

from polytopes.polytopes_utils import h_rep_minimal, find_interior, compute_path_in_polytope
from initialization.grid_utils import GSplatVoxel
from polytopes.collision_set import GSplatCollisionSet
from polytopes.decomposition import compute_polytope
from ellipsoids.intersection_utils import compute_intersection_linear_motion

class SplatPlan():
    def __init__(self, gsplat, robot_config, env_config, device):
        # gsplat: GSplat object

        self.gsplat = gsplat
        self.device = device

        # Robot configuration
        self.radius = robot_config['radius']
        self.vmax = robot_config['vmax']
        self.amax = robot_config['amax']
        self.collision_set = GSplatCollisionSet(self.gsplat, self.vmax, self.amax, self.radius, self.device)

        # Environment configuration (specifically voxel)
        self.lower_bound = env_config['lower_bound']
        self.upper_bound = env_config['upper_bound']
        self.resolution = env_config['resolution']

        tnow = time.time()
        torch.cuda.synchronize()
        self.gsplat_voxel = GSplatVoxel(self.gsplat, lower_bound=self.lower_bound, upper_bound=self.upper_bound, resolution=self.resolution, radius=self.radius, device=device)
        torch.cuda.synchronize()
        print('Time to create GSplatVoxel:', time.time() - tnow)

        # Save the mesh
        # gsplat_voxel.create_mesh(save_path=save_path)
        # gsplat.save_mesh(scene_name + '_gsplat.obj')

        # Record times
        self.times_cbf = []
        self.times_qp = []
        self.times_prune = []

    def generate_path(self, x0, xf, savepath=None):
        # Part 1: Computes the path seed using A*
        tnow = time.time()
        path = self.gsplat_voxel.create_path(x0, xf)
        print('Time to create path:', time.time() - tnow)

        # Part 2: Computes the collision set
        output = self.collision_set.compute_set(torch.tensor(path, device=self.device))
            # data = {
            #     'gaussian_ids': gaussian_ids[keep_gaussian],
            #     'A_bb': A0,
            #     'b_bb': b0,
            #     'b_bb_shrunk': b0 - self.radius,
            #     'path': path[i:i+2],
            #     'midpoint': 0.5 * (path[i] + path[i+1]),
            #     'means': means[keep_gaussian],
            #     'rots': rots[keep_gaussian],
            #     'scales': scales[keep_gaussian],
            #     'id': i
            # }

        # Part 3: Computes the polytope
        # polytopes = []
        As = []
        bs = []
        path = []
        for data in output:
            # For every single line segment, we always create a polytope at the first line segment, 
            # and then we subsequently check if future line segments are within the polytope before creating new ones.
            # TODO: !!!
            gs_ids = data['gaussian_ids']

            A_bb = data['A_bb']
            b_bb = data['b_bb_shrunk']
            segment = data['path']
            delta_x = segment[1] - segment[0]

            # print(segment[0])
            # print(segment[1])
            # print('Line segments')

            midpoint = data['midpoint']

            if len(gs_ids) == 0:
                continue
            elif len(gs_ids) == 1:
                rots = data['rots'].expand(2, -1, -1)
                scales = data['scales'].expand(2, -1)
                means = data['means'].expand(2, -1)
            else:
                rots = data['rots']
                scales = data['scales']
                means = data['means']

            intersection_output = compute_intersection_linear_motion(segment[0], delta_x, rots, scales, means, 
                                    R_B=None, S_B=self.radius, collision_type='sphere', 
                                    mode='bisection', N=10)

            # check1 = torch.einsum('bij, bjk, bkl->bil', (segment[0][None] - intersection_output['mu_A'])[..., None, :], intersection_output['Q_opt'], 
            #              (segment[0][None] - intersection_output['mu_A'])[..., None] ).squeeze()
            # check2 = torch.einsum('bij, bjk, bkl->bil', (segment[1][None] - intersection_output['mu_A'])[..., None, :], intersection_output['Q_opt'], 
            #         (segment[1][None] - intersection_output['mu_A'])[..., None] ).squeeze()
            
            # try:
            #     assert torch.all(check1 - intersection_output['K_opt'] >= -1e-4)
            #     assert torch.all(check2 - intersection_output['K_opt'] >= -1e-4)
            # except:
            #     print(f"Check failed {data['id']}", check1, check2, intersection_output['K_opt'])

            A, b, pts = compute_polytope(intersection_output['deltas'], intersection_output['Q_opt'], intersection_output['K_opt'], intersection_output['mu_A'])

            # check_boundary = torch.abs((torch.sum( A * pts, dim=-1) - b)) < 1e-4
            # try:
            #     assert check_boundary.all()
            # except:
            #     print(f"Check boundary failed {data['id']}", check_boundary)

            # The full polytope is a concatenation of the intersection polytope and the bounding box polytope
            A = torch.cat([A, A_bb], dim=0)
            b = torch.cat([b, b_bb], dim=0)

            norm_A = torch.linalg.norm(A, dim=-1, keepdims=True)
            A = A / norm_A
            b = b / norm_A.squeeze()

            #criterion = torch.all( (A @ segment.T - b[:, None]) <= 0., dim=0 )
            # print(criterion)
            # try:
            #     #print(f"Criterion {data['id']}", (A @ segment.T - b[:, None]).max(dim=0))
            #     assert criterion.all()
            # except:
            #     print(f"Criterion failed {data['id']}", criterion)
            #     #print(f"If failed, print intersection output {data['id']}", intersection_output['is_not_intersect'].all())

            # We want to prune the number of constraints in A and b in case there are redundant constraints. The midpoint should always be feasible
            # given manageability. 
            # TODO: Make sure this function doesn't have errors. If it does, somehow your midpoint is not feasible, and we may want to choose another
            # point that is feasible.

            # try:
            #     interior_pt = find_interior(A.cpu().numpy(), b.cpu().numpy())
            #     assert interior_pt is not None
            #     A, b = h_rep_minimal(A.cpu().numpy(), b.cpu().numpy(), interior_pt)

            #     #polytopes.append((A, b))
            #     As.append(torch.tensor(A, device=self.device))
            #     bs.append(torch.tensor(b, device=self.device))
            #     path.append(midpoint)
            # except:
            #     print('Interior point not found. Skipping polytope.')
            #     continue
            As.append(torch.tensor(A, device=self.device))
            bs.append(torch.tensor(b, device=self.device))
        self.save_polytope(As, bs, savepath + '_polytope.obj')
        return

    def save_polytope(self, A, b, save_path):
        # Initialize mesh object
        mesh = o3d.geometry.TriangleMesh()

        for A0, b0 in zip(A, b):
            # Transfer all tensors to numpy
            A0 = A0.cpu().numpy()
            b0 = b0.cpu().numpy()

            pt = find_interior(A0, b0)

            halfspaces = np.concatenate([A0, -b0[..., None]], axis=-1)
            hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
            qhull_pts = hs.intersections

            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
            bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
            mesh += bb_mesh
        
        success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

        return success

    # TODO: We need to make sure that we transform the u_out into the world frame from the ellipsoid frame for ellipsoid-ellipsoid
    def solve_QP(self, x, u_des):
        A, l, P, q = self.get_QP_matrices(x, u_des, minimal=True)

        tnow = time.time()
        u_out, success_flag = self.optimize_QP_clarabel(A, l, P, q)
        # print('Time to solve QP:', time.time() - tnow)
        self.times_qp.append(time.time() - tnow)

        self.solver_success = success_flag

        if success_flag:
            # return the optimal control
            u_out = torch.tensor(u_out).to(device=u_des.device, dtype=torch.float32) 
        else:
            # if not successful, just return the desired control but raise a warning
            print('Solver failed. Returning desired control.')
            u_out = u_des

        return u_out
    
    # Clarabel is a more robust, faster solver
    def optimize_QP_clarabel(self, A, l, P, q):
        n_constraints = A.shape[0]

        # Setup workspace
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)    

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, q, A, l, [clarabel.NonnegativeConeT(n_constraints)], settings)
        sol = solver.solve()

         # Check solver status
        if str(sol.status) != 'Solved':
            print(f"Solver status: {sol.status}")
            print(f"Number of iterations: {sol.iterations}")
            print('Clarabel did not solve the problem!')
            solver_success = False
            solution = None
        else:
            solver_success = True
            solution = sol.x

        return solution, solver_success

import torch
import numpy as np
from scipy import sparse
import clarabel
import time

from polytopes.polytopes_utils import h_rep_minimal, find_interior, compute_path_in_polytope
from initialization.grid_utils import GSplatVoxel
from polytopes.collision_set import GSplatCollisionSet

class SplatNav():
    def __init__(self, gsplat, robot_config):
        # gsplat: GSplat object

        self.gsplat = gsplat

        self.radius = radius
        collision_set = GSplatCollisionSet(gsplat, vmax, amax, radius, device)

        tnow = time.time()
        torch.cuda.synchronize()
        gsplat_voxel = GSplatVoxel(gsplat, lower_bound=lower_bound, upper_bound=upper_bound, resolution=resolution, radius=radius, device=device)
        torch.cuda.synchronize()
        print('Time to create GSplatVoxel:', time.time() - tnow)

        # Save the mesh
        # gsplat_voxel.create_mesh(save_path=save_path)
        # gsplat.save_mesh(scene_name + '_gsplat.obj')

        tnow = time.time()
        path = gsplat_voxel.create_path(x0, xf)
        print('Time to create path:', time.time() - tnow)

        # Visualize bounding boxes
        output = collision_set.compute_set(torch.tensor(path, device=device), save_path=scene_name)

        #TODO: Record times
        self.times_cbf = []
        self.times_qp = []
        self.times_prune = []

    def get_QP_matrices(self, x, u_des, minimal=True):
        # Computes the A and b matrices for the QP A u <= b


        P = np.eye(3)
        q = -1*u_des.cpu().numpy()

        A = A.cpu().numpy().squeeze()
        l = l.cpu().numpy()

        #We want to solve for a minimal set of constraints in the Polytope
        #Normalize
        norms = np.linalg.norm(A, axis=-1, keepdims=True)
        A = -A / norms
        l = -l / norms.squeeze()

        # Try to find minimal set of polytopes
        tnow = time.time()
        if minimal:

            # We know that the collision-less ellipsoids have CBF constraints that contain the origin. 
            # For those that are in collision, we don't know if the origin is in the polytope and we should
            # avoid trying to solve an optimization problem to find the interior. Because these constraints
            # are relatively few, we can just put them in the QP as is.
            collisionless = (h.cpu().numpy() > 0).squeeze()

            collisionless_A = A[collisionless]
            collisionless_l = l[collisionless]

            collision_A = A[~collisionless]
            collision_l = l[~collisionless]

            # print('Is Robot in Collision?: ', np.all(collisionless), 'Number of collisions:', np.sum(~collisionless))

            try:
                try:
                    # If the robot is safe, the origin should be solution (u = -(alpha + beta) v)
                    feasible_pt = -(self.alpha_constant + self.beta_constant) * x[..., 3:6].cpu().numpy()
                    Aminimal, lminimal = h_rep_minimal(collisionless_A, collisionless_l, feasible_pt)
                except:
                    # Hopefully it should never be in this case.
                    print('The origin is not a feasible point. Resorting to solving Chebyshev center for an interior point.')
                    # Find interior point through Chebyshev center
                    # feasible_pt = find_interior(A, l)
                    # Aminimal, lminimal = h_rep_minimal(A, l, feasible_pt)               
                    raise ValueError('Failed to find an interior point for the minimal polytope.')
                
                #print('Reduction in polytope size:', 1 - Aminimal.shape[0] / A.shape[0], 'Final polytope size:', Aminimal.shape[0])
                A, l = np.concatenate([Aminimal, collision_A], axis=0), np.concatenate([lminimal, collision_l], axis=0)
            except:
                print('Failed to compute minimal polytope. Keeping all constraints.')
                pass
        # print('Time to compute minimal polytope:', time.time() - tnow)
        self.times_prune.append(time.time() - tnow)

        return A, l, P, q

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

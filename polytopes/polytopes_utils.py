import numpy as np
import scipy
import torch
from scipy.optimize import linprog
import cvxpy as cvx
from ellipsoids.intersection_utils import gs_sphere_intersection_eval

# TODO: We should specify what type of method we are using (sphere or ellipsoid, and sampling or bisection)
# TODO: This function needs a big refactor and split into multiple functions and naming conventions
def compute_supporting_hyperplanes(R, D, kappa, mu_A, test_pt, tau):

    batch = R.shape[0]
    dim = R.shape[-1]

    evals = gs_sphere_intersection_eval(R, D, kappa, mu_A, test_pt, tau)

    K_j = evals[0]
    inds = evals[1]

    ss = torch.linspace(0., 1., 100, device=R.device)[1:-1]
    s_max = ss[inds]

    lambdas = D

    S_j_flat = (s_max*(1-s_max))[..., None] / (kappa + s_max[..., None] * (lambdas - kappa))

    S_j = torch.diag_embed(S_j_flat)
    A_j = torch.bmm(R, torch.bmm(S_j, R.transpose(1, 2)))

    delta_j = test_pt - mu_A

    A = -torch.bmm(delta_j.reshape(batch, 1, -1), A_j).squeeze()
    b = -torch.sqrt(K_j) + torch.sum(A*mu_A, dim=-1)

    proj_points = mu_A + delta_j / torch.sqrt(K_j)[..., None]

    return A.cpu().numpy().reshape(-1, dim), b.cpu().numpy().reshape(-1, 1), proj_points.cpu().numpy()


def compute_polytope(R, D, kappa, mu_A, test_pt, tau, A_bound, b_bound):
    # Find safe polytope in A <= b form
    A, b, _ = compute_supporting_hyperplanes(R, D, kappa, mu_A, test_pt, tau)

    # A, b = A.cpu().numpy(), b.cpu().numpy()

    dim = mu_A.shape[-1]

    # Add in the bounding poly constraints
    A = np.concatenate([A.reshape(-1, dim), A_bound.reshape(-1, dim)], axis=0)
    b = np.concatenate([b.reshape(-1, 1), b_bound.reshape(-1, 1)], axis=0)

    return A, b

def h_rep_minimal(A, b, pt):
    halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
    hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)

    # NOTE: It's possible that hs.dual_vertices errors out due to it not being to handle large number of facets. In that case, use the following code:
    try:
        minimal_Ab = halfspaces[hs.dual_vertices]
    except:
        qhull_pts = hs.intersections
        convex_hull = scipy.spatial.ConvexHull(qhull_pts, incremental=False, qhull_options=None)
        minimal_Ab = convex_hull.equations

    minimal_A = minimal_Ab[:, :-1]
    minimal_b = -minimal_Ab[:, -1]

    return minimal_A, minimal_b

def find_interior(A, b):
    # by way of Chebyshev center
    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(A.shape[0], 1))
    c = np.zeros(A.shape[1]+1)
    c[-1] = -1
    A = np.hstack((A, norm_vector))

    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))

    return res.x[:-1]

def polytopes_to_matrix(As, bs):
    A_sparse = scipy.linalg.block_diag(*As)
    b_sparse = np.concatenate(bs)

    return A_sparse, b_sparse

# TODO: This may be faster if we use proximal gradients
def check_and_project(A, b, point):
    # Check if Ax <= b

    criteria = A @ point - b
    is_valid = np.all(criteria < 0)

    if is_valid:
        return point
    else:
        # project point to nearest facet
        pt = cvx.Variable(3)
        obj = cvx.Minimize(cvx.norm(pt - point))
        constraints = [A @ pt <= b]
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver='CLARABEL')
        return pt.value

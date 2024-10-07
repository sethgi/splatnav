import numpy as np
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: ALLOW FOR BOTH UNIFORM SAMPLING OR BISECTION SEARCH !!!
### ________________________________________INTERSECTION TEST FOR ELLIPSOID-TO-ELLIPSOID____________________________________________________ ###
# In fact, we can just transform the world frame to a robot body spherical frame and then use the sphere to ellipsoid tests. 
# This is implicitly what the generalized eigenvalue problem is doing.

def generalized_eigen(A, B):
    # IMPORTANT!!! Assuming B is not batched (statedim x statedim), A is batched (batchdim x statedim x statedim)
    batch_dim = A.shape[0]
    state_dim = B.shape[0]

    # see NR section 11.0.5
    # L is a lower triangular matrix from the Cholesky decomposition of B
    L,_ = torch.linalg.cholesky_ex(B)

    L = L.reshape(-1, state_dim, state_dim).expand(batch_dim, -1, -1)

    # solve Y * L^T = A by a workaround
    # if https://github.com/tensorflow/tensorflow/issues/55371 is solved then this can be simplified
    Y = torch.transpose(torch.linalg.solve_triangular(L, torch.transpose(A, 1, 2), upper=False), 1, 2)

    # solve L * C = Y
    C = torch.linalg.solve_triangular(L, Y, upper=False)
    # solve the equivalent eigenvalue problem

    e, v_ = torch.linalg.eigh(C)

    # solve L^T * x = v, where x is the eigenvectors of the original problem
    v = torch.linalg.solve_triangular(torch.transpose(L, 1, 2), v_, upper=True)
    # # normalize the eigenvectors
    return e, v

def ellipsoid_intersection_test(Sigma_A, Sigma_B, mu_A, mu_B, tau):
    lambdas, Phi, v_squared = ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    KK = ellipsoid_K_function(lambdas, v_squared, tau)      # batchdim x Nsamples
    return ~torch.all(torch.any(KK > 1., dim=-1))

def ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B):
    lambdas, Phi = generalized_eigen(Sigma_A, Sigma_B) # eigh(Sigma_A, b=Sigma_B)
    v_squared = (torch.bmm(Phi.transpose(1, 2), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, Phi, v_squared

def ellipsoid_K_function(lambdas, v_squared, tau):
    batchdim = lambdas.shape[0]
    ss = torch.linspace(0., 1., 100, device=device)[1:-1].reshape(1, -1, 1)
    return (1./tau**2)*torch.sum(v_squared.reshape(batchdim, 1, -1)*((ss*(1.-ss))/(1.+ss*(lambdas.reshape(batchdim, 1, -1)-1.))), dim=2)

### ________________________________________INTERSECTION TEST FOR SPHERE-TO-ELLIPSOID____________________________________________________ ###
def gs_sphere_intersection_test(R, D, kappa, mu_A, mu_B, tau, return_raw=False):
    lambdas, v_squared = gs_sphere_intersection_test_helper(R, D, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    KK = gs_K_function(lambdas, v_squared, kappa, tau)      # batchdim x Nsamples

    if return_raw:
        test_result = torch.any(KK > 1., dim=-1)
    else:
        test_result = ~torch.all(torch.any(KK > 1., dim=-1))

    return test_result

def gs_sphere_intersection_test_helper(R, D, mu_A, mu_B):
    lambdas, v_squared = D, (torch.bmm(R.transpose(1, 2), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, v_squared

def gs_K_function(lambdas, v_squared, kappa, tau):
    batchdim = lambdas.shape[0]
    ss = torch.linspace(0., 1., 100, device=device)[1:-1].reshape(1, -1, 1)
    return (1./tau**2)*torch.sum(v_squared.reshape(batchdim, 1, -1)*((ss*(1.-ss))/(kappa + ss*(lambdas.reshape(batchdim, 1, -1) - kappa))), dim=2)

def gs_sphere_intersection_eval(R, D, kappa, mu_A, mu_B, tau):
    lambdas, v_squared = gs_sphere_intersection_test_helper(R, D, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    KK = gs_K_function(lambdas, v_squared, kappa, tau)      # batchdim x Nsamples
    K = torch.max(KK, dim=-1)
    return K

### ________________________________________INTERSECTION TEST FOR SPHERE-TO-ELLIPSOID (NUMPY VARIANTS)____________________________________________________ ###
# This section is just for timing and comparison purposes.
def gs_sphere_intersection_test_np(R, D, kappa, mu_A, mu_B, tau):
    tnow = time.time()
    lambdas, v_squared = gs_sphere_intersection_test_helper_np(R, D, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    print('helper:' , time.time() - tnow)
    tnow = time.time()
    KK = gs_K_function_np(lambdas, v_squared, kappa, tau)      # batchdim x Nsamples
    print('function eval:' , time.time() - tnow)

    tnow = time.time()
    test_result = ~np.all(np.any(KK > 1., axis=-1))
    print('boolean:' , time.time() - tnow)

    return test_result

def gs_sphere_intersection_test_helper_np(R, D, mu_A, mu_B):
    lambdas, v_squared = D, (np.matmul(np.transpose(R, (0, 2, 1)), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, v_squared

def gs_K_function_np(lambdas, v_squared, kappa, tau):
    batchdim = lambdas.shape[0]
    ss = np.linspace(0., 1., 100)[1:-1].reshape(1, -1, 1)
    return (1./tau**2)*np.sum(v_squared.reshape(batchdim, 1, -1)*((ss*(1.-ss))/(kappa + ss*(lambdas.reshape(batchdim, 1, -1) - kappa))), axis=2)



import numpy as np 
import cvxpy as cvx
import torch
import sympy as sym
import scipy
import time

# --------------------------------------------------------------------------------#
def b_spline_terms(t, deg):
    # terms are (K choose k)(1-t)**(K-k) * t**k
    terms = []
    for i in range(deg + 1):
        scaling = scipy.special.comb(deg, i)
        term = scaling * (1-t)**(deg - i) *t**i
        terms.append(term)

    return np.array(terms).astype(np.float32)

def b_spline_term_derivs(pts, deg, d):
    # terms are (K choose k)(1-t)**(K-k) * t**k
    terms = []
    for i in range(deg + 1):
        scaling = scipy.special.comb(deg, i)
        t = sym.Symbol('t')
        term = []
        for pt in pts:
            term.append(scaling * sym.diff((1-t)**(deg - i) *t**i, t, d).subs(t, pt))
        terms.append(np.array(term))

    return np.array(terms).astype(np.float32)

def create_time_pts(deg=8, N_sec=10, tf=1.):
    #Find coefficients for T splines, each connecting one waypoint to the next
    
    # THESE COEFFICIENTS YOU CAN STORE, SO YOU ONLY NEED TO COMPUTE THEM ONCE!
    time_pts = np.linspace(0., tf, N_sec)

    T = b_spline_terms(time_pts, deg)   #(deg + 1) x 2
    dT = b_spline_term_derivs(time_pts, deg, 1)
    ddT = b_spline_term_derivs(time_pts, deg, 2)
    dddT = b_spline_term_derivs(time_pts, deg, 3)
    ddddT = b_spline_term_derivs(time_pts, deg, 4)

    data = {
    'time_pts': T,
    'd_time_pts': dT,
    'dd_time_pts': ddT,
    'ddd_time_pts': dddT,
    'dddd_time_pts': ddddT
    }

    return data

### 
def polytopes_to_matrix(As, bs):
    A_sparse = scipy.linalg.block_diag(*As)
    b_sparse = np.concatenate(bs)

    return A_sparse, b_sparse

def get_qp_matrices(T, dT, ddT, dddT, ddddT, As, Bs, x0, xf):
    
    N_sec = len(As)
    deg = T[0].shape[0]
    w = deg*N_sec*3
    k = deg*3
    k3 = deg

    # Create cost
    Q_ = torch.eye(deg)
    off_diag = torch.stack([torch.arange(deg-1), torch.arange(deg-1)+1], dim=-1)
    Q_[off_diag[:, 0], off_diag[:, 1]] = -1
    Q_ = Q_ + Q_.T
    Q_[0, 0] = 1
    Q_[-1, -1] = 1
    Q__ = N_sec*3*[Q_]
    Q = torch.block_diag(*Q__)

    # Create inequality matrices
    A = []
    b = []

    # Create equality matrices
    C = torch.zeros((3* 4* N_sec, w))
    d = torch.zeros(C.shape[0])

    # Create cost matrix P, consisting only of jerk
    for i in range(N_sec):

        # Ax <= b
        A_ = torch.tensor(As[i])
        b_ = torch.tensor(Bs[i])
        A_x = deg*[A_[:, 0].reshape(-1, 1)]
        A_y = deg*[A_[:, 1].reshape(-1, 1)]
        A_z = deg*[A_[:, 2].reshape(-1, 1)]

        A_xs = torch.block_diag(*A_x)
        A_ys = torch.block_diag(*A_y)
        A_zs = torch.block_diag(*A_z)

        A_blck = torch.cat([A_xs, A_ys, A_zs], dim=-1)
        A.append(A_blck)
        b.extend(deg*[b_])

        # Cx = d
        if i < N_sec-1:
            pos1_cof = torch.tensor(T[i][:, -1]).reshape(1, -1)
            pos2_cof = torch.tensor(-T[i+1][:, 0]).reshape(1, -1)

            p1 = torch.block_diag(pos1_cof, pos1_cof, pos1_cof)
            p2 = torch.block_diag(pos2_cof, pos2_cof, pos2_cof)

            vel1_cof = torch.tensor(dT[i][:, -1]).reshape(1, -1)
            vel2_cof = torch.tensor(-dT[i+1][:, 0]).reshape(1, -1)

            v1 = torch.block_diag(vel1_cof, vel1_cof, vel1_cof)
            v2 = torch.block_diag(vel2_cof, vel2_cof, vel2_cof)

            acc1_cof = torch.tensor(ddT[i][:, -1]).reshape(1, -1)
            acc2_cof = torch.tensor(-ddT[i+1][:, 0]).reshape(1, -1)

            a1 = torch.block_diag(acc1_cof, acc1_cof, acc1_cof)
            a2 = torch.block_diag(acc2_cof, acc2_cof, acc2_cof)

            jer1_cof = torch.tensor(dddT[i][:, -1]).reshape(1, -1)
            jer2_cof = torch.tensor(-dddT[i+1][:, 0]).reshape(1, -1)

            j1 = torch.block_diag(jer1_cof, jer1_cof, jer1_cof)
            j2 = torch.block_diag(jer2_cof, jer2_cof, jer2_cof)

            C_t1 = torch.cat([p1, v1, a1, j1], axis=0)
            C_t2 = torch.cat([p2, v2, a2, j2], axis=0)
            C_t = torch.cat([C_t1, C_t2], axis=-1)

            n, m = C_t.shape
            n_e = m//2
            C[n*i: n*(i+1), n_e*i:n_e*(i+2)] = C_t
    
    # Create inequality matrices
    A = torch.block_diag(*A)
    b = torch.cat(b, dim=0)
    b = b.reshape((-1,))

    # Append initial and final position constraints
    p0_cof = torch.tensor(T[0][:, 0]).reshape(1, -1)
    pf_cof = torch.tensor(T[-1][:, -1]).reshape(1, -1)

    p0 = torch.block_diag(p0_cof, p0_cof, p0_cof)
    pf = torch.block_diag(pf_cof, pf_cof, pf_cof)

    C_ = torch.zeros((3*2, w))
    C_[:3, 0:n_e] = p0
    C_[3:, -n_e:] = pf

    d_ = torch.tensor(np.concatenate([x0, xf], axis=0))

    # Concatenate G and h matrices
    C = torch.cat([C, C_], axis=0)

    d = torch.cat([d, d_], axis=0)
    d = d.reshape((-1,))

    return A.cpu().numpy(), b.cpu().numpy(), C.cpu().numpy(), d.cpu().numpy(), Q.cpu().numpy()

######################################################################################################

def compute_path_from_corridor(As, bs, x0, xf):
    # Compute path from union of polytopes
    # TODO: Add in spline support

    num_pts = len(As)

    points = cvx.Variable((num_pts, 3))
    
    cost = cvx.pnorm(points[:-1] - points[1:])
    obj = cvx.Minimize(cost)

    A, b = polytopes_to_matrix(As, bs)
    constraints = [A @ cvx.reshape(points, num_pts*3, order='C') <= b]

    prob = cvx.Problem(obj, constraints)
    prob.solve()
    # prob.solve()

    solved_pts = points.value

    if solved_pts is not None:
        traj = np.concatenate([x0.reshape(1, 3), solved_pts.reshape(-1, 3), xf.reshape(1, 3)], axis=0)
        return traj
    else:
        return None
    
class SplinePlanner():
    def __init__(self, spline_deg=6, N_sec=10) -> None:
        self.spline_deg = spline_deg
    
        ### Create the time points matrix/coefficients for the Bezier curve
        self.time_pts = create_time_pts(deg=spline_deg, N_sec=N_sec)

    def optimize_one_step(self, A, b, x0, xf):
        tnow = time.time()
        self.calculate_b_spline_coeff_one_step(A, b, x0, xf)
        print('opt:', time.time() - tnow)

        tnow = time.time()
        output = self.eval_b_spline()
        print('eval:', time.time() - tnow)

        return output

    def calculate_b_spline_coeff_one_step(self, A, b, x0, xf):
        N_sections = len(A)         #Number of segments

        T = self.time_pts['time_pts']
        dT = self.time_pts['d_time_pts']
        ddT = self.time_pts['dd_time_pts']
        dddT = self.time_pts['ddd_time_pts']
        ddddT = self.time_pts['dddd_time_pts']

        # Copy time points N times
        T_list = [T]*N_sections
        dT_list = [dT]*N_sections
        ddT_list = [ddT]*N_sections
        dddT_list = [dddT]*N_sections
        ddddT_list = [ddddT]*N_sections

        #Set up CVX problem
        A_prob, b_prob, C_prob, d_prob, Q_prob = get_qp_matrices(T_list, dT_list, ddT_list, dddT_list, ddddT_list, A, b, x0, xf)
        
        # eliminate endpoint constraint
        C_prob = C_prob[:-3]
        d_prob = d_prob[:-3]
        
        n_var = C_prob.shape[-1]

        x = cvx.Variable(n_var)

        final_point = cvx.reshape(x, (N_sections*3, -1), order='C')[-3:, -1]

        obj = cvx.Minimize(cvx.quad_form(x, Q_prob)) # + cvx.norm(final_point - xf))

        constraints = [A_prob @ x <= b_prob, C_prob @ x == d_prob]

        prob = cvx.Problem(obj, constraints)

        prob.solve(solver='CLARABEL')
        
        coeffs = []
        cof_splits = np.split(x.value, N_sections)
        for cof_split in cof_splits:
            xyz = np.split(cof_split, 3)
            cof = np.stack(xyz, axis=0)
            coeffs.append(cof)

        self.coeffs = np.array(coeffs)
        return self.coeffs, prob.value

    def optimize_b_spline(self, As, Bs, x0, xf):
        self.calculate_b_spline_coeff(As, Bs, x0, xf)
        return self.eval_b_spline()

    def calculate_b_spline_coeff(self, As, Bs, x0, xf):
        N_sections = len(As)         #Number of segments

        T = self.time_pts['time_pts']
        dT = self.time_pts['d_time_pts']
        ddT = self.time_pts['dd_time_pts']
        dddT = self.time_pts['ddd_time_pts']
        ddddT = self.time_pts['dddd_time_pts']

        # Copy time points N times
        T_list = [T]*N_sections
        dT_list = [dT]*N_sections
        ddT_list = [ddT]*N_sections
        dddT_list = [dddT]*N_sections
        ddddT_list = [ddddT]*N_sections

        #Set up CVX problem
        A_prob, b_prob, C_prob, d_prob, Q_prob = get_qp_matrices(T_list, dT_list, ddT_list, dddT_list, ddddT_list, As, Bs, x0, xf)
        n_var = C_prob.shape[-1]

        x = cvx.Variable(n_var)

        obj = cvx.Minimize(cvx.quad_form(x, Q_prob))

        constraints = [A_prob @ x <= b_prob, C_prob @ x == d_prob]

        prob = cvx.Problem(obj, constraints)

        prob.solve(solver='CLARABEL')
        
        coeffs = []
        cof_splits = np.split(x.value, N_sections)
        for cof_split in cof_splits:
            xyz = np.split(cof_split, 3)
            cof = np.stack(xyz, axis=0)
            coeffs.append(cof)

        self.coeffs = np.array(coeffs)
        return self.coeffs, prob.value

    def eval_b_spline(self):
        T = self.time_pts['time_pts']
        dT = self.time_pts['d_time_pts']
        ddT = self.time_pts['dd_time_pts']
        dddT = self.time_pts['ddd_time_pts']
        ddddT = self.time_pts['dddd_time_pts']

        full_traj = []
        for coeff in self.coeffs:
            pos = (coeff @ T).T
            vel = (coeff @ dT).T
            acc = (coeff @ ddT).T
            jerk = (coeff @ dddT).T
            sub_traj = np.concatenate([pos, vel, acc, jerk], axis=-1)
            full_traj.append(sub_traj)

        return np.concatenate(full_traj, axis=0)
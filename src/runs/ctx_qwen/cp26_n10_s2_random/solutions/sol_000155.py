# sol_000155 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000141 (state d8f6c168) state=1d52432b sum of radii=2.630957 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
A_ub_pre = None
pair_indices = None

def get_lp_matrices():
    """Pre-construct the sparse structure of the LP constraint matrix."""
    num_pairs = N * (N - 1) // 2
    num_bound = 4 * N
    A = np.zeros((num_pairs + num_bound, N))
    pairs = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            pairs.append((i, j))
            idx += 1
    for i in range(N):
        for _ in range(4):
            A[idx, i] = 1.0
            idx += 1
    return A, pairs

# Initialize LP structure globally
A_ub_pre, pair_indices = get_lp_matrices()

def solve_lp_and_gradient(centers):
    """
    Solves LP for optimal radii given fixed centers and computes the gradient 
    of the sum of radii with respect to center positions using LP duals.
    """
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    b_ub = np.zeros(A_ub_pre.shape[0])
    idx = 0
    for i, j in pair_indices:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_ub_pre, b_ub=b_ub, 
                  bounds=[(0, u) for u in ub], method='highs')
    if not res.success:
        return None, None, None
        
    radii = res.x
    duals = res.ineqlin.marginals
    
    grad = np.zeros_like(centers)
    idx = 0
    for i, j in pair_indices:
        lam = duals[idx]
        if lam > 1e-8:
            d = dists[i, j]
            vec = (centers[i] - centers[j]) / d
            grad[i] += lam * vec
            grad[j] -= lam * vec
        idx += 1
        
    boundary_start = len(pair_indices)
    for i in range(n):
        grad[i, 0] += duals[boundary_start + 4*i] - duals[boundary_start + 4*i + 1]
        grad[i, 1] += duals[boundary_start + 4*i + 2] - duals[boundary_start + 4*i + 3]
        
    return radii, -res.fun, grad

def lbfgs_val_grad(v):
    """Objective and gradient wrapper for L-BFGS-B."""
    centers = v.reshape(N, 2)
    centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
    _, val, grad = solve_lp_and_gradient(centers)
    if val is None:
        return 0.0, np.zeros_like(v)
    return -val, -grad.reshape(-1)

def obj_joint(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint SLSQP: boundary and pairwise non-overlap (squared for smoothness)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    idx = np.triu_indices(N, 1)
    dx = c[idx[0], 0] - c[idx[1], 0]
    dy = c[idx[0], 1] - c[idx[1], 1]
    con = np.concatenate([con, dx**2 + dy**2 - (r[idx[0]] + r[idx[1]])**2])
    return con

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.085, 0.095, 0.105, 0.115]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        starts.append(c)
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_sum = -1.0
    bounds_xy = [(0.0, 1.0)] * (2 * N)
    
    starts = generate_starts(rng)
    
    # Phase 1: L-BFGS-B optimization on centers using exact LP gradient
    for c0 in starts:
        v0 = c0.flatten()
        try:
            res = minimize(lbfgs_val_grad, v0, jac=True,
                          method='L-BFGS-B', bounds=bounds_xy,
                          options={'maxiter': 800, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            _, s_opt, _ = solve_lp_and_gradient(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation search from best to escape local minima
    for _ in range(40):
        c_trial = best_centers.copy()
        c_trial += rng.normal(0, 0.006, c_trial.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        v_trial = c_trial.flatten()
        try:
            res = minimize(lbfgs_val_grad, v_trial, jac=True,
                          method='L-BFGS-B', bounds=bounds_xy,
                          options={'maxiter': 400, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            _, s_opt, _ = solve_lp_and_gradient(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
        except Exception:
            pass
            
    # Phase 3: SLSQP joint polish on centers and radii
    r_lp, _, _ = solve_lp_and_gradient(best_centers)
    v0 = np.concatenate([best_centers.flatten(), r_lp])
    bounds_joint = bounds_xy + [(0.0, 0.5)] * N
    
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
        if np.sum(res.x[2*N:]) > best_sum:
            best_centers = res.x[:2*N].reshape(N, 2)
            best_radii = res.x[2*N:]
        else:
            best_radii = r_lp
    except Exception:
        best_radii = r_lp
        
    # Phase 4: Strict numerical repair to guarantee validation passes
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                req = best_radii[i] + best_radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    best_radii[i] -= shrink
                    best_radii[j] -= shrink
                    changed = True
        for i in range(N):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            mr = min(x, 1.0-x, y, 1.0-y)
            if r > mr + 1e-12:
                best_radii[i] = mr
                changed = True
        if not changed:
            break
            
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(np.sum(best_radii))

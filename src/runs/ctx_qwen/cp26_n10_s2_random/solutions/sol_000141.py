# sol_000141 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000119 (state ab7c4e6b) state=d8f6c168 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def get_lp_matrices(n):
    """Pre-construct the sparse structure of the LP constraint matrix."""
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A_ub = np.zeros((num_pairs + num_bound, n))
    pair_indices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pair_indices.append((i, j))
            idx += 1
            
    for i in range(n):
        for _ in range(4):
            A_ub[idx, i] = 1.0
            idx += 1
            
    return A_ub, pair_indices

# Precompute LP matrix structure globally
A_ub_pre, pair_indices = get_lp_matrices(N)

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
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
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
        
    c_obj = -np.ones(n)
    bounds_r = [(0, u) for u in ub]
    
    res = linprog(c_obj, A_ub=A_ub_pre, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return None, None, None
        
    radii = res.x
    duals = res.ineqlin.marginals
    
    grad = np.zeros_like(centers)
    
    # Pairwise forces from distance constraints
    idx = 0
    for i, j in pair_indices:
        lam = duals[idx]
        if lam > 1e-7:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    # Boundary forces from wall constraints
    boundary_start = len(pair_indices)
    for i in range(n):
        mu_L = duals[boundary_start + 4*i]
        mu_R = duals[boundary_start + 4*i + 1]
        mu_B = duals[boundary_start + 4*i + 2]
        mu_T = duals[boundary_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, -res.fun, grad

def optimize_gradient(centers0, steps=1500, init_step=0.008):
    """Runs gradient ascent on centers to maximize sum of radii."""
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    
    step = init_step
    no_improve = 0
    
    for k in range(steps):
        radii, curr_sum, grad = solve_lp_and_gradient(centers)
        if radii is None:
            break
            
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        # Adaptive step size decay
        if no_improve > 40:
            step *= 0.6
        elif no_improve > 15:
            step *= 0.8
            
        if step < 1e-8:
            break
            
        centers += step * grad
        centers = np.clip(centers, 0.01, 0.99)
        
        # Occasional jitter to escape local minima
        if k % 300 == 0 and k > 0:
            centers += np.random.normal(0, 0.0015, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            
    return best_centers, best_sum

def generate_starts(n, rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 6], [6, 6, 4, 5, 5], [5, 6, 4, 5, 6]
    ]
    
    for pat in patterns:
        for r_est in [0.09, 0.095, 0.10]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3)
            c = np.array(c[:n])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # Add purely random starts
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (n, 2))
        starts.append(c)
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_sum = -1.0
    
    # Phase 1: Gradient ascent from multiple starts
    starts = generate_starts(N, rng)
    for c0 in starts:
        c_opt, s_opt = optimize_gradient(c0)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            
    # Phase 2: SLSQP polish on the best configuration
    def obj_joint(v): return -np.sum(v[2*N:])
    
    def cons_joint(v):
        c = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
        idx = np.triu_indices(N, 1)
        d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
        con.append(d - (r[idx[0]] + r[idx[1]]))
        return np.concatenate(con)
        
    r_lp, _, _ = solve_lp_and_gradient(best_centers)
    v0 = np.concatenate([best_centers.flatten(), r_lp])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 5000, 'ftol': 1e-14})
        if np.sum(res.x[2*N:]) > best_sum:
            best_centers = res.x[:2*N].reshape(N, 2)
            best_radii = res.x[2*N:]
        else:
            best_radii = r_lp
    except Exception:
        best_radii = r_lp
        
    # Phase 3: Strict numerical repair to guarantee validation passes
    for _ in range(100):
        changed = False
        # Fix overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                req = best_radii[i] + best_radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    best_radii[i] -= shrink
                    best_radii[j] -= shrink
                    changed = True
        # Fix boundary violations
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

# sol_000184 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000143 (state c665f9c9) state=af413dce sum of radii=2.621304 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import math
import warnings

warnings.filterwarnings('ignore')

N = 26

def get_lp_structure(n):
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

# Global LP structure
A_LP, PAIR_IDX = get_lp_structure(N)

def solve_lp_and_grad(centers):
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
    
    b_ub = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in PAIR_IDX:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    bounds_r = [(0, u) for u in ub]
    
    res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return np.full(n, 1e-9), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract duals (marginals) depending on scipy version
    duals = None
    if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
        duals = res.ineqlin.marginals
    else:
        duals = np.zeros_like(b_ub)
        
    grad = np.zeros_like(centers)
    
    idx = 0
    for i, j in PAIR_IDX:
        lam = duals[idx]
        if lam > 1e-7:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    boundary_start = len(PAIR_IDX)
    for i in range(n):
        mu_L = duals[boundary_start + 4*i]
        mu_R = duals[boundary_start + 4*i + 1]
        mu_B = duals[boundary_start + 4*i + 2]
        mu_T = duals[boundary_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, np.sum(radii), grad

def generate_hex_start(pat, r0, rng):
    """Generates an initial configuration from a perturbed hexagonal lattice."""
    c = []
    y = r0
    for r_idx, cnt in enumerate(pat):
        shift = r0 if r_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(c) < N:
                c.append([x, y])
            x += 2.0 * r0
        y += r0 * math.sqrt(3)
    c = np.array(c[:N])
    c += rng.normal(0, 0.003, c.shape)
    return np.clip(c, 0.05, 0.95)

def generate_force_start(rng):
    """Generates an initial configuration using repulsive forces."""
    c = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(600):
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 0.18 and d > 1e-6:
                    push = (0.18 - d) * 0.08
                    forces[i] += (diff / d) * push
                    forces[j] -= (diff / d) * push
        c += forces
        c = np.clip(c, 0.05, 0.95)
    return c

def optimize_gradient_ascent(centers0, max_iter=1500, init_step=0.006, rng=None):
    """Runs gradient ascent on centers to maximize sum of radii."""
    if rng is None:
        rng = np.random.default_rng(42)
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    
    radii, curr_sum, grad = solve_lp_and_grad(centers)
    if radii is None:
        return centers0, -1.0
    best_sum = curr_sum
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            centers += step * grad / g_norm
        else:
            step *= 0.5
            
        centers = np.clip(centers, 0.001, 0.999)
        
        radii, curr_sum, grad = solve_lp_and_grad(centers)
        if radii is None:
            break
            
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            step = min(step * 1.12, 0.025)
        else:
            step *= 0.90
            
        # Periodic jitter to escape local minima
        if k > 0 and k % 200 == 0:
            noise_scale = 0.004 * (0.7 ** (k // 200))
            centers += rng.normal(0, noise_scale, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            radii, curr_sum, grad = solve_lp_and_grad(centers)
            
        if step < 1e-7:
            break
            
    return best_centers, best_sum

def objective_joint(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def constraints_joint(v):
    """Computes boundary and pairwise non-overlap constraints."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    idx_i, idx_j = np.triu_indices(N, 1)
    d = np.linalg.norm(c[idx_i] - c[idx_j], axis=1)
    con.append(d - (r[idx_i] + r[idx_j]))
    return np.concatenate(con)

def repair_solution(centers, radii):
    """Deterministic repair to guarantee validation passes."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-12
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            mr = min(x, 1.0 - x, y, 1.0 - y)
            if r > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing():
    """Packs 26 circles in a unit square to maximize sum of radii."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_sum = -1.0
    best_radii = None
    
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 6], [6, 6, 4, 5, 5], [5, 6, 4, 5, 6],
        [4, 5, 5, 6, 6], [6, 6, 6, 4, 4], [5, 7, 5, 5, 4],
        [4, 6, 5, 6, 5], [5, 6, 6, 4, 5]
    ]
    
    # Phase 1: Diverse starts + Gradient Ascent
    for pat in patterns:
        for r0 in [0.088, 0.092, 0.096, 0.100, 0.104]:
            c0 = generate_hex_start(pat, r0, rng)
            c_opt, s_opt = optimize_gradient_ascent(c0, max_iter=1200, init_step=0.006, rng=rng)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                
    for _ in range(10):
        c0 = generate_force_start(rng)
        c_opt, s_opt = optimize_gradient_ascent(c0, max_iter=1200, init_step=0.006, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            
    for _ in range(8):
        c0 = rng.uniform(0.15, 0.85, (N, 2))
        c_opt, s_opt = optimize_gradient_ascent(c0, max_iter=1000, init_step=0.005, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            
    # Phase 2: SLSQP Joint Polish
    if best_centers is not None:
        r_lp, _, _ = solve_lp_and_grad(best_centers)
        if r_lp is None:
            r_lp = np.full(N, 0.05)
            
        v0 = np.concatenate([best_centers.flatten(), r_lp])
        bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        
        try:
            res = minimize(objective_joint, v0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints_joint},
                           options={'maxiter': 15000, 'ftol': 1e-14})
            if np.sum(res.x[2 * N:]) > best_sum:
                best_centers = res.x[:2 * N].reshape(N, 2)
                best_radii = res.x[2 * N:]
                best_sum = np.sum(best_radii)
            else:
                best_radii = r_lp
        except Exception:
            best_radii = r_lp
            
        # Phase 3: Coordinate-wise local search / Perturbation refinement
        for _ in range(50):
            c_try = best_centers.copy()
            idx = rng.integers(N)
            c_try[idx] += rng.normal(0, 0.006, 2)
            c_try = np.clip(c_try, 0.01, 0.99)
            
            r_try, s_try, _ = solve_lp_and_grad(c_try)
            if r_try is not None and s_try > best_sum + 1e-8:
                best_sum = s_try
                best_centers = c_try.copy()
                
        # Final gradient ascent on refined center
        c_opt, s_opt = optimize_gradient_ascent(best_centers, max_iter=800, init_step=0.004, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            r_lp, _, _ = solve_lp_and_grad(best_centers)
            if r_lp is not None:
                best_radii = r_lp
    else:
        best_radii = np.full(N, 0.05)
        
    # Phase 4: Strict numerical repair
    best_radii = repair_solution(best_centers, best_radii)
    
    return best_centers, best_radii, float(np.sum(best_radii))

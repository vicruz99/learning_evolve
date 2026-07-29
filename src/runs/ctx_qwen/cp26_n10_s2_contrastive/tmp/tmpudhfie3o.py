import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute constant constraint matrix structure for LP
A_ub_lp = np.zeros((N_PAIRS, N))
A_ub_lp[np.arange(N_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(N_PAIRS), J_IDX] = 1.0

def objective(x):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints:
    - Pairwise distance >= sum of radii
    - Circle boundaries within [0,1]x[0,1]
    Returns array of constraint values (must be >= 0)
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Vectorized pairwise distance constraints
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_dist = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_b = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_dist, c_b])

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def polish(centers, radii, iters=8000):
    """Run SLSQP to jointly optimize centers and radii from a starting point."""
    x0 = np.zeros(3 * N)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = np.maximum(radii * 0.99, 1e-5)
    
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                       constraints=cons_opt, options={'maxiter': iters, 'ftol': 1e-13, 'disp': False})
        c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
        r_opt = solve_lp_radii(c_opt)
        return c_opt, r_opt
    except Exception:
        return centers, radii

def generate_inits(rng):
    """Generate diverse initial center configurations."""
    inits = []
    # Hexagonal lattices with varying spacings and offsets
    for sp in np.linspace(0.15, 0.22, 9):
        for seed in range(6):
            rng_local = np.random.RandomState(seed + int(sp * 1000))
            c = np.zeros((N, 2))
            idx = 0
            row = 0
            y = sp / 2
            while idx < N and y < 1.0 - sp / 2:
                x_start = sp / 2 + (row % 2) * sp / 2
                col = 0
                while x_start + col * sp < 1.0 - sp / 2 and idx < N:
                    c[idx, 0] = x_start + col * sp
                    c[idx, 1] = y
                    idx += 1
                    col += 1
                y += sp * np.sqrt(3) / 2
                row += 1
            while idx < N:
                c[idx] = rng_local.uniform(0.1, 0.9, 2)
                idx += 1
            c += rng_local.normal(0, 0.005, c.shape)
            c = np.clip(c, 0.02, 0.98)
            inits.append(c)
            
    # Random uniform placements
    for _ in range(40):
        c = rng.uniform(0.05, 0.95, (N, 2))
        inits.append(c)
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = 0.0
    best_c = None
    best_r = None
    
    inits = generate_inits(rng)
    
    # Stage 1: Broad multi-start search
    for c0 in inits:
        r0 = solve_lp_radii(c0)
        c_p, r_p = polish(c0, r0, iters=6000)
        s = np.sum(r_p)
        if s > best_sum:
            best_sum = s
            best_c = c_p.copy()
            best_r = r_p.copy()
            
    # Stage 2: Basin hopping with decaying noise
    if best_c is not None:
        cur_c = best_c.copy()
        cur_r = best_r.copy()
        cur_s = best_sum
        
        for step in range(250):
            noise_scale = 0.025 * np.exp(-step / 70.0)
            c_pert = cur_c + rng.normal(0, noise_scale, cur_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > cur_s:
                cur_c, cur_r, cur_s = c_pert, r_pert, s_pert
                c_pol, r_pol = polish(c_pert, r_pert, iters=5000)
                s_pol = np.sum(r_pol)
                if s_pol > cur_s:
                    cur_c, cur_r, cur_s = c_pol, r_pol, s_pol
                    
                if cur_s > best_sum:
                    best_sum = cur_s
                    best_c = cur_c.copy()
                    best_r = cur_r.copy()
                    
    # Stage 3: Fine local search
    if best_c is not None:
        for scale in [0.006, 0.003, 0.001]:
            for _ in range(40):
                c_pert = best_c + rng.normal(0, scale, best_c.shape)
                c_pert = np.clip(c_pert, 0.01, 0.99)
                r_pert = solve_lp_radii(c_pert)
                s_pert = np.sum(r_pert)
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_c = c_pert.copy()
                    best_r = r_pert.copy()
                    c_pol, r_pol = polish(c_pert, r_pert, iters=4000)
                    if np.sum(r_pol) > best_sum:
                        best_sum = np.sum(r_pol)
                        best_c = c_pol.copy()
                        best_r = r_pol.copy()

    # Fallback safety net
    if best_c is None:
        best_c = inits[0]
        best_r = solve_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
    # Strict post-processing to guarantee validator compliance
    centers = best_c.copy()
    radii = best_r.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], 
                 centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(0.0, radii[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                if d < radii[i] + radii[j] - 1e-11:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
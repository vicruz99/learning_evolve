import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute LP constraint matrix for efficiency
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
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def make_hex_init(spacing, seed, margin=0.02):
    """Generate a hexagonal lattice initialization with controlled noise."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin + spacing / 2
    while idx < N and y < 1.0 - margin - spacing / 2:
        x_start = margin + spacing / 2 + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - margin - spacing / 2 and idx < N:
            centers[idx, 0] = x_start + col * spacing
            centers[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin, 1.0 - margin, 2)
        idx += 1
    centers += rng.normal(0, 0.003, centers.shape)
    return np.clip(centers, 0.01, 0.99)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Diverse Initializations & SLSQP Refinement
    inits = []
    # Hexagonal lattices with varying spacing
    for sp in np.linspace(0.13, 0.21, 12):
        for seed in range(8):
            inits.append(make_hex_init(sp, seed))
            
    # Square grids
    for sp in np.linspace(0.14, 0.20, 8):
        for seed in range(5):
            rng_g = np.random.RandomState(seed + 100)
            c = np.zeros((N, 2))
            idx = 0
            y = sp/2
            while y < 1.0 - sp/2 and idx < N:
                x = sp/2
                while x < 1.0 - sp/2 and idx < N:
                    c[idx] = [x, y] + rng_g.normal(0, 0.002, 2)
                    idx += 1
                    x += sp
                y += sp
            while idx < N:
                c[idx] = rng_g.uniform(0.1, 0.9, 2)
                idx += 1
            inits.append(np.clip(c, 0.02, 0.98))
            
    # Random placements
    for _ in range(40):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for init_c in inits:
        r_init, s_init = solve_lp_radii(init_c)
        if s_init == 0: continue
        x0 = np.zeros(3*N)
        x0[0::3] = init_c[:, 0]
        x0[1::3] = init_c[:, 1]
        x0[2::3] = np.maximum(r_init * 0.95, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt, s_opt = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Simulated Annealing on Centers with LP evaluation
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r, curr_s = solve_lp_radii(curr_c)
        best_c = curr_c.copy()
        best_r = curr_r.copy()
        best_s = curr_s
        
        for step in range(5000):
            temp = 0.02 * np.exp(-step / 1200.0)
            scale = temp
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            delta = s_pert - curr_s
            if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-6)):
                curr_c = c_pert
                curr_r = r_pert
                curr_s = s_pert
                
                if curr_s > best_s:
                    best_s = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                    
                    # Local SLSQP polish
                    x0 = np.zeros(3*N)
                    x0[0::3] = best_c[:, 0]
                    x0[1::3] = best_c[:, 1]
                    x0[2::3] = np.maximum(best_r * 0.98, 1e-5)
                    try:
                        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                        c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_opt, s_opt = solve_lp_radii(c_opt)
                        if s_opt > best_s:
                            best_s = s_opt
                            best_c = c_opt.copy()
                            best_r = r_opt.copy()
                            curr_c = c_opt
                            curr_r = r_opt
                            curr_s = s_opt
                    except:
                        pass
                        
        best_centers = best_c
        best_radii = best_r
        best_sum = best_s

    # Phase 3: Fine perturbations to escape shallow minima
    if best_centers is not None:
        for _ in range(150):
            c_fin = best_centers + rng.normal(0, 0.0008, best_centers.shape)
            c_fin = np.clip(c_fin, 0.01, 0.99)
            r_fin, s_fin = solve_lp_radii(c_fin)
            if s_fin > best_sum:
                best_sum = s_fin
                best_centers = c_fin.copy()
                best_radii = r_fin.copy()
                
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                   options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    c_f = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_f, s_f = solve_lp_radii(c_f)
                    if s_f > best_sum:
                        best_sum = s_f
                        best_centers = c_f.copy()
                        best_radii = r_f.copy()
                except:
                    pass
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = make_hex_init(0.17, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Strict post-processing to guarantee validator compliance
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(0.0, radii[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))
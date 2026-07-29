import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure
A_ub_lp = np.zeros((N_PAIRS, N))
A_ub_lp[np.arange(N_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(N_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + N_PAIRS)
    c[:N] = cx - r
    c[N:2 * N] = 1.0 - cx - r
    c[2 * N:3 * N] = cy - r
    c[3 * N:4 * N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4 * N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def generate_hex_init(seed, spacing, margin=0.05):
    """Generate hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin + spacing * np.sqrt(3) / 2
    while idx < N and y < 1.0 - margin:
        x_start = margin + (row % 2) * spacing / 2.0
        col = 0
        while x_start + col * spacing < 1.0 - margin and idx < N:
            centers[idx, 0] = x_start + col * spacing
            centers[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin, 1.0 - margin, 2)
        idx += 1
    centers += rng.normal(0, 0.004, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Diverse initial configurations
    inits = []
    for sp in np.linspace(0.14, 0.22, 9):
        for s in range(5):
            inits.append(generate_hex_init(s, sp))
    for s in range(30):
        inits.append(rng.uniform(0.08, 0.92, (N, 2)))
        
    # Phase 1a: Broad search
    for init_c in inits:
        r_init, _ = solve_lp_radii(init_c)
        r_init = np.maximum(r_init * 0.96, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = init_c[:, 0]
        x0[1::3] = init_c[:, 1]
        x0[2::3] = r_init
        
        for method in ['SLSQP', 'trust-constr']:
            try:
                res = minimize(objective, x0, method=method, bounds=bounds_opt,
                               constraints=cons_opt,
                               options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt, s_opt = solve_lp_radii(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
            except Exception:
                pass
                
    # Phase 2: Basin hopping with adaptive noise
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(250):
            # Adaptive decaying noise
            scale = 0.018 * np.exp(-step / 60.0) + 0.001
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > 0:
                # Simulated annealing acceptance criterion
                accept = s_pert > curr_s or rng.random() < np.exp((s_pert - curr_s) / (0.005 + 0.01*np.exp(-step/50.0)))
                if accept:
                    curr_c = c_pert
                    curr_r = r_pert
                    curr_s = s_pert
                    
                    if s_pert > best_sum:
                        # Polish new best
                        x0 = np.zeros(3 * N)
                        x0[0::3] = curr_c[:, 0]
                        x0[1::3] = curr_c[:, 1]
                        x0[2::3] = np.maximum(curr_r * 0.97, 1e-5)
                        try:
                            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                           constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                            c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                            r_pol, s_pol = solve_lp_radii(c_pol)
                            if s_pol > best_sum:
                                best_sum = s_pol
                                best_centers = c_pol.copy()
                                best_radii = r_pol.copy()
                                curr_c, curr_r, curr_s = c_pol, r_pol, s_pol
                        except Exception:
                            pass

    # Phase 3: Targeted subset perturbations (1-3 circles at a time)
    if best_centers is not None:
        for _ in range(400):
            num_pert = rng.choice([1, 2, 3])
            idxs = rng.choice(N, num_pert, replace=False)
            c_pert = best_centers.copy()
            c_pert[idxs] += rng.normal(0, 0.003, (num_pert, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                
                x0 = np.zeros(3 * N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.97, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                    c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_pol, s_pol = solve_lp_radii(c_pol)
                    if s_pol > best_sum:
                        best_sum = s_pol
                        best_centers = c_pol.copy()
                        best_radii = r_pol.copy()
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        best_centers = generate_hex_init(0, 0.18)
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Strict post-processing to guarantee validator compliance
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(0.0, radii[i])
        
    # Iteratively resolve any remaining numerical overlaps conservatively
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(best_centers[i,0] - best_centers[j,0], 
                               best_centers[i,1] - best_centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    overlap = radii[i] + radii[j] - d
                    # Remove half overlap from each, preserving sum as much as possible
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))
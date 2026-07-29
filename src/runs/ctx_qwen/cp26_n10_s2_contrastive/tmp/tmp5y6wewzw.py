import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c_overlap = np.hypot(cx[I_IDX] - cx[J_IDX], cy[I_IDX] - cy[J_IDX]) - (r[I_IDX] + r[J_IDX])
    c_bound = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_overlap, c_bound])

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(N_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def generate_init_centers(seed, style='hex', sp=None):
    """Generate diverse initial center configurations."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    
    if style == 'hex':
        s = sp if sp is not None else 0.16 + rng.uniform(-0.02, 0.06)
        idx = 0
        row = 0
        y = s / 2
        while idx < N and y < 1.0 - s / 2:
            x_start = s / 2 + (row % 2) * s / 2
            col = 0
            while x_start + col * s < 1.0 - s / 2 and idx < N:
                centers[idx, 0] = x_start + col * s
                centers[idx, 1] = y
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
    elif style == 'grid':
        s = sp if sp is not None else 0.19 + rng.uniform(-0.02, 0.02)
        idx = 0
        y = s / 2
        while idx < N and y < 1.0 - s / 2:
            x = s / 2
            while x < 1.0 - s / 2 and idx < N:
                centers[idx, 0] = x
                centers[idx, 1] = y
                idx += 1
                x += s
            y += s
    elif style == 'pattern':
        patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,6,6,5,5], [8,6,5,4,3]]
        pat = patterns[seed % len(patterns)]
        y = 0.05
        dy = 0.9 / (len(pat) - 0.5)
        idx = 0
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.08
            x = 0.05 + shift
            for _ in range(cnt):
                if idx < N:
                    centers[idx, 0] = x
                    centers[idx, 1] = y
                    idx += 1
                x += 0.17
            y += dy
    else:
        centers = rng.uniform(0.1, 0.9, (N, 2))
        
    centers += rng.normal(0, 0.008, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple:
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
    
    # Phase 1: Diverse initializations
    inits = []
    for sp in np.linspace(0.14, 0.22, 10):
        for seed in range(5):
            inits.append((generate_init_centers(seed, 'hex', sp), 0.0))
    for seed in range(10):
        inits.append((generate_init_centers(seed, 'grid'), 0.0))
    for seed in range(10):
        inits.append((generate_init_centers(seed, 'pattern'), 0.0))
    for seed in range(15):
        inits.append((rng.uniform(0.1, 0.9, (N, 2)), 0.0))
        
    for init_c, noise in inits:
        c0 = init_c + rng.normal(0, noise, init_c.shape)
        c0 = np.clip(c0, 0.02, 0.98)
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0 * 0.95, 1e-5)
        
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt, s_opt = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Basin hopping with decaying noise
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(200):
            scale = 0.012 * np.exp(-step / 60.0)
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > 0:
                x0 = np.zeros(3*N)
                x0[0::3] = c_pert[:, 0]
                x0[1::3] = c_pert[:, 1]
                x0[2::3] = np.maximum(r_pert * 0.98, 1e-5)
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                   options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
                    c_new = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new, s_new = solve_lp_radii(c_new)
                    
                    if s_new > curr_s:
                        curr_s = s_new
                        curr_c = c_new.copy()
                        curr_r = r_new.copy()
                        if s_new > best_sum:
                            best_sum = s_new
                            best_centers = c_new.copy()
                            best_radii = r_new.copy()
                except Exception:
                    pass
                    
    # Phase 3: Local perturbation of few circles to escape shallow minima
    if best_centers is not None:
        for _ in range(150):
            n_pert = rng.randint(1, 4)
            idxs = rng.choice(N, n_pert, replace=False)
            c_pert = best_centers.copy()
            c_pert[idxs] += rng.normal(0, 0.003, (n_pert, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.98, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                   options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                    c_new = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new, s_new = solve_lp_radii(c_new)
                    if s_new > best_sum:
                        best_sum = s_new
                        best_centers = c_new.copy()
                        best_radii = r_new.copy()
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        best_centers = generate_init_centers(0)
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
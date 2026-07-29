import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute constant structure for LP pairwise constraints
A_LP = np.zeros((N_PAIRS, N))
A_LP[np.arange(N_PAIRS), I_IDX] = 1.0
A_LP[np.arange(N_PAIRS), J_IDX] = 1.0

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
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

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

def relax_centers(centers, radii, steps=15):
    """Deterministically push overlapping circles apart to improve packing feasibility."""
    c = centers.copy()
    r = radii.copy()
    n = c.shape[0]
    for _ in range(steps):
        moves = np.zeros_like(c)
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = math.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-12:
                    overlap = r[i] + r[j] - d
                    shift = overlap / 2.0
                    ux = dx / d
                    uy = dy / d
                    moves[i] += np.array([ux * shift, uy * shift])
                    moves[j] -= np.array([ux * shift, uy * shift])
        c += moves * 0.5
        c = np.clip(c, 1e-6, 1.0 - 1e-6)
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattices with varying spacings and offsets
    for sp in np.linspace(0.14, 0.21, 10):
        for seed in range(6):
            c = np.zeros((N, 2))
            idx = 0
            row = 0
            y = sp / 2
            rng_s = np.random.RandomState(seed * 1000 + int(sp * 10000))
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
                c[idx] = rng_s.uniform(0.1, 0.9, 2)
                idx += 1
            c += rng_s.normal(0, 0.005, c.shape)
            c = np.clip(c, 0.02, 0.98)
            inits.append(c)
            
    # 2. Structured row patterns
    patterns = [[6,7,7,6], [7,6,6,7], [8,7,6,5], [5,7,7,7], [6,6,6,8], [9,7,7,3], [7,7,7,5], [5,6,6,5,4], [6,5,5,6,4]]
    for pat in patterns:
        c = np.zeros((N, 2))
        idx = 0
        y = 0.05
        dy = 0.9 / len(pat)
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.05
            x = 0.05 + shift
            step = (0.9 - 2 * shift) / cnt if cnt > 0 else 1.0
            for _ in range(cnt):
                if idx < N:
                    c[idx] = [x, y]
                    idx += 1
                x += step
            y += dy
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        c += rng.normal(0, 0.008, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    # 3. Random uniform placements
    for _ in range(30):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # Phase 1: Broad multi-start search
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0 * 0.95, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt, s_opt = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Adaptive Basin Hopping & Relaxation
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(250):
            # Decaying noise schedule
            noise_scale = 0.014 * np.exp(-step / 60.0)
            c_pert = curr_c + rng.normal(0, noise_scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            # Relax to clear overlaps before LP evaluation
            c_relaxed = relax_centers(c_pert, curr_r, steps=10)
            r_pert, s_pert = solve_lp_radii(c_relaxed)
            
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_relaxed, r_pert, s_pert
                
                # Local SLSQP polishing after successful jump
                x0_p = np.zeros(3 * N)
                x0_p[0::3] = curr_c[:, 0]
                x0_p[1::3] = curr_c[:, 1]
                x0_p[2::3] = np.maximum(curr_r * 0.98, 1e-5)
                
                try:
                    res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                    c_opt = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                    r_opt, s_opt = solve_lp_radii(c_opt)
                    if s_opt > curr_s:
                        curr_c, curr_r, curr_s = c_opt, r_opt, s_opt
                        
                except Exception:
                    pass
                    
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
                    
            # Occasional larger jumps to escape deep basins
            if step % 30 == 0 and step > 0:
                c_jump = curr_c + rng.normal(0, 0.035, curr_c.shape)
                c_jump = np.clip(c_jump, 0.05, 0.95)
                r_jump, s_jump = solve_lp_radii(c_jump)
                if s_jump > curr_s:
                    curr_c, curr_r, curr_s = c_jump, r_jump, s_jump
                    
    # Phase 3: Fine local perturbations to squeeze remaining gaps
    if best_centers is not None:
        for scale in [0.006, 0.003, 0.0015, 0.0008]:
            for _ in range(40):
                c_fin = best_centers + rng.normal(0, scale, best_centers.shape)
                c_fin = np.clip(c_fin, 0.01, 0.99)
                r_fin, s_fin = solve_lp_radii(c_fin)
                if s_fin > best_sum:
                    x0 = np.zeros(3 * N)
                    x0[0::3] = c_fin[:, 0]
                    x0[1::3] = c_fin[:, 1]
                    x0[2::3] = np.maximum(r_fin * 0.99, 1e-5)
                    try:
                        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                       constraints=cons_opt, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                        c_f = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_f, s_f = solve_lp_radii(c_f)
                        if s_f > best_sum:
                            best_sum = s_f
                            best_centers = c_f.copy()
                            best_radii = r_f.copy()
                    except Exception:
                        continue

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    centers = best_centers.copy()
    radii = best_radii.copy()
    
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
                d = math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(N_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective_joint(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    dists = np.hypot(dx, dy)
    c_overlap = dists - (r[I_IDX] + r[J_IDX])
    
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    return np.concatenate([c_overlap, c_bound])

def make_strictly_feasible(centers, radii):
    """Deterministically resolve overlaps and boundary violations."""
    c = centers.copy()
    r = radii.copy()
    for i in range(N):
        ub = min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
        if r[i] > ub:
            r[i] = max(0.0, ub - 1e-9)
            
    for _ in range(150):
        changed = False
        for k in range(N_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            if d < r[i] + r[j] - 1e-12:
                exc = r[i] + r[j] - d
                r[i] -= exc * 0.5
                r[j] -= exc * 0.5
                changed = True
        if not changed:
            break
    return c, np.maximum(r, 0.0)

def generate_hex_init(seed, rows_pattern=None, noise=0.008):
    """Generate structured initial center configurations."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 2))
    idx = 0
    
    if rows_pattern is None:
        sp = 0.17 + rng.uniform(-0.02, 0.02)
        margin = 0.04 + rng.uniform(-0.01, 0.01)
        y = margin
        row = 0
        while idx < N and y < 1.0 - margin:
            x = margin + (row % 2) * sp / 2.0
            while x < 1.0 - margin and idx < N:
                centers[idx] = [x, y]
                x += sp
                idx += 1
            y += sp * np.sqrt(3) / 2.0
            row += 1
    else:
        y = 0.05
        dy = 0.90 / (len(rows_pattern) - 0.5)
        for r_idx, cnt in enumerate(rows_pattern):
            shift = 0.0 if r_idx % 2 == 0 else 0.09
            x = 0.05 + shift
            step = (0.90 - 2*shift) / (cnt - 0.5) if cnt > 1 else 0.0
            for _ in range(cnt):
                if idx < N:
                    centers[idx] = [x, y]
                    idx += 1
                x += step
            y += dy
            
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
        
    centers += rng.normal(0, noise, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def polish_joint(c0, r0):
    """Run SLSQP to jointly optimize centers and radii from a starting point."""
    r_safe = np.maximum(r0 * 0.95, 1e-5)
    x0 = np.zeros(3 * N)
    x0[0::3] = c0[:, 0]
    x0[1::3] = c0[:, 1]
    x0[2::3] = r_safe
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_joint}
    
    try:
        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        co = np.column_stack((res.x[0::3], res.x[1::3]))
        ro, s = solve_lp_radii(co)
        return co, ro, s
    except Exception:
        return c0, r0, np.sum(r0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Phase 1: Diverse Initializations
    inits = []
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,6,5,4,3], [5,5,5,5,6], [6,6,4,5,5]]
    for s in range(25):
        inits.append(generate_hex_init(s))
    for pat in patterns:
        inits.append(generate_hex_init(rng.integers(1000), rows_pattern=pat))
    for s in range(30):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Stage 1: Initial Polish
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        co, ro, s = polish_joint(c0, r0)
        if s > best_sum:
            best_sum = s
            best_c = co.copy()
            best_r = ro.copy()
            
    # Stage 2: Basin Hopping with LP evaluation
    if best_c is not None:
        cur_c = best_c.copy()
        cur_r = best_r.copy()
        cur_s = best_sum
        temp = 0.02
        
        for step in range(400):
            temp = 0.02 * np.exp(-step / 100.0)
            noise_scale = 0.012 * np.sqrt(max(temp, 1e-4))
            c_pert = cur_c + rng.normal(0, noise_scale, cur_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            if s_pert > cur_s or rng.random() < np.exp((s_pert - cur_s) / max(temp, 1e-4)):
                cur_c = c_pert
                cur_r = r_pert
                cur_s = s_pert
                
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_c = cur_c.copy()
                    best_r = cur_r.copy()
                    co, ro, s_polished = polish_joint(best_c, best_r)
                    if s_polished > best_sum:
                        best_sum = s_polished
                        best_c = co.copy()
                        best_r = ro.copy()
                        
    # Stage 3: Fine local search
    if best_c is not None:
        for scale in [0.008, 0.004, 0.002, 0.001]:
            for _ in range(25):
                c_pert = best_c + rng.normal(0, scale, best_c.shape)
                c_pert = np.clip(c_pert, 0.01, 0.99)
                r_pert, s_pert = solve_lp_radii(c_pert)
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_c = c_pert.copy()
                    best_r = r_pert.copy()
                    co, ro, s_polished = polish_joint(best_c, best_r)
                    if s_polished > best_sum:
                        best_sum = s_polished
                        best_c = co.copy()
                        best_r = ro.copy()
                        
    # Fallback safety net
    if best_c is None:
        best_c = inits[0]
        best_r, best_sum = solve_lp_radii(best_c)
        
    # Strict post-processing to guarantee validator compliance
    best_c, best_r = make_strictly_feasible(best_c, best_r)
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
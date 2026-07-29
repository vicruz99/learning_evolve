import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure for speed
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
        bounds.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
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
    
    c = np.empty(4 * N + N_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    dists = np.hypot(dx, dy)
    c[4*N:] = dists - (r[I_IDX] + r[J_IDX])
    return c

cons_joint = {'type': 'ineq', 'fun': constraints_joint}
bounds_joint = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N

def relax_centers(centers, radii, steps=30, strength=0.5):
    """Deterministically push overlapping circles apart to improve packing density."""
    c = centers.copy()
    r = radii.copy()
    n = c.shape[0]
    for _ in range(steps):
        moves = np.zeros_like(c)
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-12:
                    overlap = r[i] + r[j] - d
                    shift = overlap / 2.0
                    ux = dx / d
                    uy = dy / d
                    moves[i] += np.array([ux * shift, uy * shift])
                    moves[j] -= np.array([ux * shift, uy * shift])
        c += moves * strength
        c = np.clip(c, 1e-6, 1.0 - 1e-6)
    return c

def generate_hex_init(seed, rows_pattern=None, noise_scale=0.008):
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
        dy = 0.9 / (len(rows_pattern) - 0.5)
        for r_idx, cnt in enumerate(rows_pattern):
            shift = 0.0 if r_idx % 2 == 0 else 0.09
            x = 0.05 + shift
            for _ in range(cnt):
                if idx < N:
                    centers[idx] = [x, y]
                    idx += 1
                x += (0.9 - 2*shift) / max(cnt - 1, 1)
            y += dy
            
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
        
    centers += rng.normal(0, noise_scale, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def polish_joint(c0):
    """Run SLSQP to jointly optimize centers and radii from a starting point."""
    r0 = solve_lp_radii(c0)[0] * 0.95
    r0 = np.maximum(r0, 1e-5)
    x0 = np.zeros(3*N)
    x0[0::3] = c0[:, 0]
    x0[1::3] = c0[:, 1]
    x0[2::3] = r0
    try:
        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_joint,
                       constraints=cons_joint, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        co = res.x[:2*N].reshape(N, 2)
        ro, s = solve_lp_radii(co)
        return co, ro, s
    except Exception:
        ro, s = solve_lp_radii(c0)
        return c0, ro, s

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = 0.0
    best_c = None
    best_r = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    for s in range(25):
        inits.append(generate_hex_init(s))
        
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,6,5,4,3], [5,5,5,5,6], [6,6,4,5,5]]
    for pat in patterns:
        inits.append(generate_hex_init(rng.integers(1000), rows_pattern=pat, noise_scale=0.005))
        
    for s in range(30):
        c = rng.uniform(0.15, 0.85, (N, 2))
        inits.append(c)
        
    # Stage 1: Initial polish of all starts using LP + Joint SLSQP
    for c0 in inits:
        co, ro, s = polish_joint(c0)
        if s > best_sum:
            best_sum = s
            best_c = co.copy()
            best_r = ro.copy()
            
    # Stage 2: Basin hopping on centers with LP evaluation & relaxation
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
            
            # Relax to push apart overlaps before LP evaluation
            c_relaxed = relax_centers(c_pert, cur_r, steps=20, strength=0.4)
            r_pert, s_pert = solve_lp_radii(c_relaxed)
            
            # Simulated annealing acceptance
            if s_pert > cur_s or rng.random() < np.exp((s_pert - cur_s) / max(temp, 1e-4)):
                cur_c = c_relaxed
                cur_r = r_pert
                cur_s = s_pert
                
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_c = cur_c.copy()
                    best_r = cur_r.copy()
                    # Polish new best with SLSQP
                    co, ro, s_polished = polish_joint(best_c)
                    if s_polished > best_sum:
                        best_sum = s_polished
                        best_c = co.copy()
                        best_r = ro.copy()
                        
    # Stage 3: Fine local search & targeted perturbations
    if best_c is not None:
        for scale in [0.006, 0.003, 0.0015, 0.0008]:
            for _ in range(30):
                c_pert = best_c + rng.normal(0, scale, best_c.shape)
                c_pert = np.clip(c_pert, 0.01, 0.99)
                r_pert, s_pert = solve_lp_radii(c_pert)
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_c = c_pert.copy()
                    best_r = r_pert.copy()
                    co, ro, s_polished = polish_joint(best_c)
                    if s_polished > best_sum:
                        best_sum = s_polished
                        best_c = co.copy()
                        best_r = ro.copy()
                        
    # Fallback safety net
    if best_c is None:
        best_c = inits[0]
        best_r, best_sum = solve_lp_radii(best_c)
        
    # Strict post-processing to guarantee validator compliance
    c_final = best_c.copy()
    r_final = best_r.copy()
    
    # Enforce boundaries strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-10)
        r_final[i] = max(r_final[i], 0.0)
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = c_final[i, 0] - c_final[j, 0]
                dy = c_final[i, 1] - c_final[j, 1]
                d = np.hypot(dx, dy)
                if d < r_final[i] + r_final[j] - 1e-12:
                    exc = r_final[i] + r_final[j] - d
                    r_final[i] -= exc * 0.5
                    r_final[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return c_final, r_final, float(np.sum(r_final))
# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000094 (state 4cf54399) state=185d6fba sum of radii=2.620598 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure for speed
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

bounds_opt = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    c_obj = -np.ones(N)
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(N):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 1e-6)

def relax_centers(centers, radii, steps=50):
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
        c += moves * 0.5
        c = np.clip(c, 1e-6, 1.0 - 1e-6)
    return c

def objective_joint(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * N:])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    c = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    
    b = np.empty(4 * N + NUM_PAIRS)
    b[:N] = c[:, 0] - r
    b[N:2 * N] = 1.0 - c[:, 0] - r
    b[2 * N:3 * N] = c[:, 1] - r
    b[3 * N:4 * N] = 1.0 - c[:, 1] - r
    
    dx = c[I_IDX, 0] - c[J_IDX, 0]
    dy = c[I_IDX, 1] - c[J_IDX, 1]
    dists = np.hypot(dx, dy)
    b[4 * N:] = dists - (r[I_IDX] + r[J_IDX])
    return b

cons_joint = {'type': 'ineq', 'fun': constraints_joint}

def generate_hex_init(seed, rows_pattern=None):
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
        y = 0.06
        dy = 0.88 / (len(rows_pattern) - 0.5)
        for r_idx, cnt in enumerate(rows_pattern):
            shift = 0.0 if r_idx % 2 == 0 else 0.09
            x = 0.06 + shift
            for _ in range(cnt):
                if idx < N:
                    centers[idx] = [x, y]
                    idx += 1
                x += (0.88 - 2 * shift) / (cnt - 0.5) if cnt > 1 else 0.0
            y += dy
            
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
        
    centers += rng.normal(0, 0.008, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def polish(c0):
    """Run SLSQP to jointly optimize centers and radii from a starting point."""
    r0 = solve_lp_radii(c0) * 0.95
    r0 = np.maximum(r0, 1e-5)
    x0 = np.concatenate([c0.flatten(), r0])
    try:
        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                       constraints=cons_joint, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
        co = res.x[:2 * N].reshape(N, 2)
        ro = solve_lp_radii(co)
        s = np.sum(ro)
        return co, ro, s
    except Exception:
        ro = solve_lp_radii(c0)
        return c0, ro, np.sum(ro)

def run_packing():
    best_sum = 0.0
    best_c = None
    best_r = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    for s in range(30):
        inits.append(generate_hex_init(s))
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,6,5,4,3], [5,5,5,5,6], [6,6,4,5,5]]
    for pat in patterns:
        inits.append(generate_hex_init(rng.integers(1000), rows_pattern=pat))
    for s in range(20):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Stage 1: Initial polish of all starts
    for c0 in inits:
        co, ro, s = polish(c0)
        if s > best_sum:
            best_sum = s
            best_c = co.copy()
            best_r = ro.copy()
            
    # Stage 2: Basin hopping with relaxation and LP evaluation
    if best_c is not None:
        cur_c = best_c.copy()
        cur_r = best_r.copy()
        cur_s = best_sum
        temp = 0.02
        
        for step in range(300):
            temp = 0.02 * np.exp(-step / 80.0)
            noise_scale = 0.015 * np.sqrt(max(temp, 1e-4))
            c_pert = cur_c + rng.normal(0, noise_scale, cur_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            # Relax to push apart overlaps before LP evaluation
            c_relaxed = relax_centers(c_pert, cur_r, steps=20)
            r_pert = solve_lp_radii(c_relaxed)
            s_pert = np.sum(r_pert)
            
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
                    co, ro, s_polished = polish(best_c)
                    if s_polished > best_sum:
                        best_sum = s_polished
                        best_c = co.copy()
                        best_r = ro.copy()
                        
    # Stage 3: Fine local search
    if best_c is not None:
        for scale in [0.008, 0.004, 0.002, 0.001]:
            for _ in range(20):
                c_pert = best_c + rng.normal(0, scale, best_c.shape)
                c_pert = np.clip(c_pert, 0.01, 0.99)
                r_pert = solve_lp_radii(c_pert)
                s_pert = np.sum(r_pert)
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_c = c_pert.copy()
                    best_r = r_pert.copy()
                    co, ro, s_polished = polish(best_c)
                    if s_polished > best_sum:
                        best_sum = s_polished
                        best_c = co.copy()
                        best_r = ro.copy()
                        
    # Fallback safety net
    if best_c is None:
        best_c = inits[0]
        best_r = solve_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
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
    for _ in range(100):
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

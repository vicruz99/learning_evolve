# sol_000127 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000082 (state 4b2dee7c) state=abefc104 sum of radii=2.418359 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)
NUM_PAIRS = len(I)
A_LP = np.zeros((NUM_PAIRS, N))
A_LP[np.arange(NUM_PAIRS), I] = 1.0
A_LP[np.arange(NUM_PAIRS), J] = 1.0

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(N):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(N), 0.0

def generate_inits():
    """Generate diverse structured and random initial center configurations."""
    inits = []
    rng = np.random.RandomState(42)
    
    # Hexagonal lattices with varying spacing and alignment
    for s in np.linspace(0.18, 0.24, 12):
        for shift in [0.0, s / 2.0]:
            c = np.zeros((N, 2))
            idx = 0
            y = s / 2.0
            row = 0
            while idx < N and y < 1.0 - s / 2.0:
                x = s / 2.0 + shift + (row % 2) * s / 2.0
                while x < 1.0 - s / 2.0 and idx < N:
                    c[idx] = [x, y]
                    x += s
                    idx += 1
                y += s * np.sqrt(3) / 2.0
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            inits.append(c + rng.normal(0, 0.005, c.shape))
            
    # Structured row patterns optimized for 26 circles
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,5,5,5,3], [6,6,5,5,4], [5,5,5,5,6]]
    for pat in patterns:
        c = np.zeros((N, 2))
        idx = 0
        y = 0.06
        dy = 0.88 / len(pat)
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.04
            x = 0.06 + shift
            step_x = (0.88 - 2 * shift) / cnt
            for _ in range(cnt):
                if idx < N:
                    c[idx] = [x, y]
                    x += step_x
                idx += 1
            y += dy
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        inits.append(c + rng.normal(0, 0.006, c.shape))
        
    # Random feasible placements
    for _ in range(30):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def obj_func(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def cons_func(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx, cy, r = x[0::3], x[1::3], x[2::3]
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    c_ov = dx**2 + dy**2 - (r[I] + r[J])**2
    
    c_bd = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_ov, c_bd])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    inits = generate_inits()
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Phase 1: Broad search with LP evaluation
    for c0 in inits:
        c0 = np.clip(c0, 0.02, 0.98)
        r0, s0 = solve_radii_lp(c0)
        if s0 > best_sum:
            best_sum = s0
            best_c = c0.copy()
            best_r = r0.copy()
            
    # Phase 2: Basin hopping / Local search on centers using LP oracle
    rng = np.random.RandomState(123)
    current_c = best_c.copy()
    current_r = best_r.copy()
    current_s = best_sum
    
    step = 0.012
    for epoch in range(1500):
        noise = rng.normal(0, step, (N, 2))
        c_new = current_c + noise
        c_new = np.clip(c_new, 0.01, 0.99)
        
        r_new, s_new = solve_radii_lp(c_new)
        
        if s_new > current_s:
            current_c = c_new
            current_r = r_new
            current_s = s_new
            if current_s > best_sum:
                best_sum = current_s
                best_c = current_c.copy()
                best_r = current_r.copy()
            step = min(step * 1.02, 0.04)
        else:
            step = max(step * 0.998, 0.0005)
            
        # Occasional large jump to escape local minima
        if epoch % 200 == 199:
            c_jump = best_c + rng.normal(0, 0.025, (N, 2))
            c_jump = np.clip(c_jump, 0.02, 0.98)
            r_jump, s_jump = solve_radii_lp(c_jump)
            if s_jump > best_sum:
                best_sum = s_jump
                best_c = c_jump.copy()
                best_r = r_jump.copy()
                current_c = best_c
                current_r = best_r
                current_s = best_sum
                
    # Phase 3: SLSQP joint polish to extract remaining microscopic gains
    x0 = np.zeros(3 * N)
    x0[0::3] = best_c[:, 0]
    x0[1::3] = best_c[:, 1]
    x0[2::3] = np.maximum(best_r * 0.98, 1e-5)
    
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    try:
        res = minimize(obj_func, x0, method='SLSQP', bounds=bounds_opt, 
                       constraints={'type': 'ineq', 'fun': cons_func},
                       options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
        if res.success:
            cx = res.x[0::3]
            cy = res.x[1::3]
            c_polish = np.column_stack((cx, cy))
            r_polish, s_polish = solve_radii_lp(c_polish)
            if s_polish > best_sum:
                best_sum = s_polish
                best_c = c_polish
                best_r = r_polish
    except Exception:
        pass
        
    # Phase 4: Strict post-processing to guarantee validity
    centers = best_c.copy()
    radii = best_r.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], 
                 centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(radii[i], 0.0)
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc / 2.0
                    radii[j] -= exc / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum

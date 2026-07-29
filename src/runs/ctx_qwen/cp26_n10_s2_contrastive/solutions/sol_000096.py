# sol_000096 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000090 (state 81009fa6) state=58f551d8 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIRS_I)
A_ub = np.zeros((NUM_PAIRS, N))
A_ub[np.arange(NUM_PAIRS), PAIRS_I] = 1.0
A_ub[np.arange(NUM_PAIRS), PAIRS_J] = 1.0

def solve_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    dx = centers[PAIRS_I, 0] - centers[PAIRS_J, 0]
    dy = centers[PAIRS_I, 1] - centers[PAIRS_J, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(N):
        mx, my = centers[i]
        ub = min(mx, 1.0-mx, my, 1.0-my)
        bounds.append((0.0, max(1e-8, ub)))
        
    try:
        res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    try:
        res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    rng = np.random.default_rng(42)
    best_c, best_r, best_s = None, None, 0.0
    
    inits = []
    # 1. Hexagonal lattices with varying spacing
    for sp in np.linspace(0.18, 0.24, 6):
        c = np.zeros((N, 2))
        idx = 0; y = sp/2; row = 0
        while idx < N and y < 1.0 - sp/2:
            x = sp/2 + (row%2)*sp/2
            while x < 1.0 - sp/2 and idx < N:
                c[idx] = [x, y]; idx += 1; x += sp
            y += sp * np.sqrt(3)/2; row += 1
        while idx < N: c[idx] = [0.5, 0.5]; idx += 1
        inits.append(c)
        
    # 2. Random uniform placements
    for s in range(30):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # 3. Corner/Edge biased configurations
    for s in range(20):
        c = rng.uniform(0.1, 0.9, (N, 2))
        c[:4] = [[0.1,0.1], [0.9,0.1], [0.1,0.9], [0.9,0.9]]
        inits.append(c)
        
    # Evaluate all initial configurations
    for c0 in inits:
        c0 = np.clip(c0, 0.02, 0.98)
        r0 = solve_lp(c0)
        s0 = np.sum(r0)
        if s0 > best_s:
            best_s, best_c, best_r = s0, c0.copy(), r0.copy()
            
    c_curr = best_c.copy()
    s_curr = best_s
    
    # Phase 1: Broad Basin Hopping
    for step in range(3000):
        noise = 0.015 * (0.999 ** step)
        c_pert = np.clip(c_curr + rng.normal(0, noise, (N, 2)), 0.01, 0.99)
        r_pert = solve_lp(c_pert)
        s_pert = np.sum(r_pert)
        
        if s_pert > s_curr:
            c_curr, s_curr = c_pert, s_pert
            if s_curr > best_s:
                best_s, best_c, best_r = s_curr, c_curr.copy(), r_pert.copy()
        elif rng.random() < 0.03:
            c_curr, s_curr = c_pert, s_pert
            
    # Phase 2: Fine Local Search (Hill Climbing)
    c_curr = best_c.copy()
    s_curr = best_s
    for step in range(4000):
        noise = 0.002 * (0.998 ** step)
        c_pert = np.clip(c_curr + rng.normal(0, noise, (N, 2)), 0.01, 0.99)
        r_pert = solve_lp(c_pert)
        s_pert = np.sum(r_pert)
        if s_pert > s_curr:
            c_curr, s_curr = c_pert, s_pert
            if s_curr > best_s:
                best_s, best_c, best_r = s_curr, c_curr.copy(), r_pert.copy()
                
    # Strict post-processing to guarantee validator compliance
    c_final = best_c.copy()
    r_final = best_r.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx, my = c_final[i]
        ub = min(mx, 1.0-mx, my, 1.0-my)
        r_final[i] = min(r_final[i], ub - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(50):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = PAIRS_I[k], PAIRS_J[k]
            d = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed: break
        
    return c_final, r_final, float(np.sum(r_final))

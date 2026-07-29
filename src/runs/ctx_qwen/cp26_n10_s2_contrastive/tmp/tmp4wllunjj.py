import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure for speed
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(N):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(-np.ones(N), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    try:
        res = linprog(-np.ones(N), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 0.01)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def sa_optimize(init_c, rng, steps=1200):
    """Simulated annealing on centers maximizing LP radii sum."""
    c = init_c.copy()
    r = solve_lp_radii(c)
    curr_sum = np.sum(r)
    best_c = c.copy()
    best_r = r.copy()
    best_sum = curr_sum
    
    temp = 0.015
    for i in range(steps):
        step_size = 0.004 * (1.0 - i/steps) + 0.0005
        c_new = c + rng.normal(0, step_size, c.shape)
        c_new = np.clip(c_new, 0.01, 0.99)
        
        r_new = solve_lp_radii(c_new)
        sum_new = np.sum(r_new)
        
        delta = sum_new - curr_sum
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-4)):
            c = c_new
            r = r_new
            curr_sum = sum_new
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c = c.copy()
                best_r = r.copy()
        temp *= 0.999
        
    return best_c, best_r, best_sum

def jiggle_optimize(centers, rng, iters=400):
    """Deterministic coordinate-descent style polishing of centers."""
    c = centers.copy()
    r = solve_lp_radii(c)
    curr_sum = np.sum(r)
    best_c = c.copy()
    best_r = r.copy()
    best_sum = curr_sum
    
    dirs = np.array([[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]])
    scale = 0.002
    
    for _ in range(iters):
        idx = rng.integers(N)
        moved = False
        for d in dirs:
            c_try = c.copy()
            c_try[idx] += d * scale
            c_try = np.clip(c_try, 0.01, 0.99)
            r_try = solve_lp_radii(c_try)
            if np.sum(r_try) > curr_sum:
                c = c_try
                r = r_try
                curr_sum = np.sum(r_try)
                moved = True
                break
        if not moved:
            scale *= 0.85
        else:
            scale *= 1.05
            
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_c = c.copy()
            best_r = r.copy()
            
    return best_c, best_r, best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations (hexagonal lattices + random)
    inits = []
    for seed in range(10):
        r_gen = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        s = 0.14 + r_gen.uniform(0, 0.08)
        y = s/2
        row = 0
        while idx < N and y < 1.0 - s/2:
            x_start = s/2 + (row % 2) * s/2
            col = 0
            while x_start + col*s < 1.0 - s/2 and idx < N:
                c[idx] = [x_start + col*s, y]
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(0.1, 0.9, 2)
            idx += 1
        c += r_gen.normal(0, 0.01, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    for _ in range(5):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Phase 1: Simulated Annealing on Centers
    for c0 in inits:
        bc, br, bs = sa_optimize(c0, rng, steps=1200)
        if bs > best_sum:
            best_sum = bs
            best_c = bc
            best_r = br
            
    # Phase 2: Deterministic Jiggle Refinement
    if best_c is not None:
        bc, br, bs = jiggle_optimize(best_c, rng, iters=400)
        if bs > best_sum:
            best_sum = bs
            best_c = bc
            best_r = br
            
    # Phase 3: Joint SLSQP Polish
    if best_c is not None:
        x0 = np.zeros(3*N)
        x0[0::3] = best_c[:, 0]
        x0[1::3] = best_c[:, 1]
        x0[2::3] = best_r * 0.98
        
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = solve_lp_radii(c_opt)
                if np.sum(r_opt) > best_sum:
                    best_c = c_opt
                    best_r = r_opt
                    best_sum = np.sum(r_opt)
        except Exception:
            pass
            
    # Fallback safety net
    if best_c is None:
        best_c = inits[0]
        best_r = solve_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    centers = best_c.copy()
    radii = best_r.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], 
                 centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max(0.0, mx - 1e-9))
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            if d < radii[i] + radii[j] - 1e-11:
                exc = radii[i] + radii[j] - d
                radii[i] -= exc * 0.5
                radii[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
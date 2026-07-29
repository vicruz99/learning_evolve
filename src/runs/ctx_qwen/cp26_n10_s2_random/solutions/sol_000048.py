# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000044 (state 69bc282d) state=14317d24 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26

def evaluate_constraints(p):
    """
    Computes all inequality constraint values (must be >= 0).
    Layout: [x1, y1, r1, x2, y2, r2, ...]
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    # Boundary constraints: circles inside [0,1]x[0,1]
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
        r  # non-negative radii
    ])
    
    # Overlap constraints: squared distance >= squared sum of radii
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    rsq = (r[:, None] + r[None, :])**2
    
    iu, ju = np.triu_indices(N, k=1)
    c_overlap = dist_sq[iu, ju] - rsq[iu, ju]
    
    return np.concatenate([c_bound, c_overlap])

def objective(p):
    """Objective: minimize negative sum of radii."""
    return -np.sum(p[2::3])

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-6, 0.5]"""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    return b

def init_hex(r0=0.095, noise=0.0):
    """Generates a hexagonal lattice initialization."""
    p = np.zeros(N * 3)
    idx = 0
    y = r0
    row = 0
    while idx < N:
        x = r0 if row % 2 == 0 else 2 * r0
        while x + r0 <= 1.0 + 1e-9 and idx < N:
            p[3 * idx] = x
            p[3 * idx + 1] = y
            p[3 * idx + 2] = r0
            idx += 1
            x += 2 * r0
        y += np.sqrt(3) * r0
        row += 1
        
    if noise > 0:
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        p[0::3] += rng.normal(0, noise, N)
        p[1::3] += rng.normal(0, noise, N)
        p[0::3] = np.clip(p[0::3], 1e-4, 1 - 1e-4)
        p[1::3] = np.clip(p[1::3], 1e-4, 1 - 1e-4)
    return p

def init_grid(r0=0.085, noise=0.0):
    """Generates a square grid initialization."""
    p = np.zeros(N * 3)
    idx = 0
    rows = 5
    cols = 6
    for r in range(rows):
        for c in range(cols):
            if idx >= N:
                break
            x = r0 + c * 2 * r0
            y = r0 + r * 2 * r0
            p[3 * idx] = x
            p[3 * idx + 1] = y
            p[3 * idx + 2] = r0
            idx += 1
            
    if noise > 0:
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        p[0::3] += rng.normal(0, noise, N)
        p[1::3] += rng.normal(0, noise, N)
        p[0::3] = np.clip(p[0::3], 1e-4, 1 - 1e-4)
        p[1::3] = np.clip(p[1::3], 1e-4, 1 - 1e-4)
    return p

def init_random(seed, r0=0.07):
    """Generates a random initialization."""
    rng = np.random.default_rng(seed)
    p = np.zeros(N * 3)
    p[0::3] = rng.uniform(0.15, 0.85, N)
    p[1::3] = rng.uniform(0.15, 0.85, N)
    p[2::3] = r0
    return p

def repair(p):
    """
    Repairs a solution by strictly enforcing constraints.
    Returns (x, y, r) arrays.
    """
    x = p[0::3].copy()
    y = p[1::3].copy()
    r = p[2::3].copy()
    
    # Enforce boundary limits
    for i in range(N):
        max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if r[i] > max_r:
            r[i] = max_r
            
    # Resolve overlaps by proportionally shrinking radii
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - 1e-12:
                    overlap = r[i] + r[j] - d
                    total = r[i] + r[j]
                    if total > 1e-12:
                        r[i] -= overlap * (r[i] / total)
                        r[j] -= overlap * (r[j] / total)
                    else:
                        r[i] -= overlap * 0.5
                        r[j] -= overlap * 0.5
                    changed = True
        if not changed:
            break
            
    r = np.maximum(r, 1e-9)
    return x, y, r

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Main optimization routine."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': evaluate_constraints}
    
    best_p = None
    best_sum = -1.0
    
    # Phase 1: Multi-start optimization with diverse configurations
    starts = []
    for r_val in [0.095, 0.09, 0.085, 0.08, 0.075]:
        for noise in [0.0, 0.005, 0.01, 0.02]:
            starts.append(init_hex(r_val, noise))
            starts.append(init_grid(r_val, noise))
            
    for seed in range(30):
        starts.append(init_random(seed, 0.07))
        
    for p0 in starts:
        # Break symmetry with tiny deterministic perturbation
        p0[0::3] += np.random.normal(0, 1e-5, N)
        p0[1::3] += np.random.normal(0, 1e-5, N)
        
        try:
            res = opt.minimize(
                objective, p0, method='SLSQP', bounds=bounds,
                constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13}
            )
            
            if res.success or res.nit > 500:
                x, y, r = repair(res.x)
                cur_sum = np.sum(r)
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_p = res.x.copy()
                    best_p[0::3] = x
                    best_p[1::3] = y
                    best_p[2::3] = r
        except Exception:
            continue
            
    # Phase 2: Local refinement around the best solution found
    if best_p is not None:
        for _ in range(15):
            p_pert = best_p.copy()
            p_pert[0::3] += np.random.normal(0, 0.003, N)
            p_pert[1::3] += np.random.normal(0, 0.003, N)
            p_pert[0::3] = np.clip(p_pert[0::3], 1e-4, 1 - 1e-4)
            p_pert[1::3] = np.clip(p_pert[1::3], 1e-4, 1 - 1e-4)
            
            try:
                res = opt.minimize(
                    objective, p_pert, method='SLSQP', bounds=bounds,
                    constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13}
                )
                x, y, r = repair(res.x)
                cur_sum = np.sum(r)
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_p = res.x.copy()
                    best_p[0::3] = x
                    best_p[1::3] = y
                    best_p[2::3] = r
            except Exception:
                pass
                
    # Fallback
    if best_p is None:
        best_p = init_hex(0.09)
        
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3]
    
    return centers, radii, float(np.sum(radii))

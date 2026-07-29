# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000043 (state e63f418f) state=2c1a60b6 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26
i_idx, j_idx = np.triu_indices(N, k=1)
PAIR_COUNT = len(i_idx)

def get_bounds():
    """Creates variable bounds: x,y in [0,1], r in [1e-6, 0.5]"""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    return b

def objective(p):
    """Minimize negative sum of radii to maximize total radius."""
    return -np.sum(p[2::3])

def constraints(p):
    """
    Computes all boundary and non-overlap constraints.
    Returns a 1D array where each element must be >= 0.
    Uses squared distances for better gradient conditioning.
    """
    x, y, r = p[0::3], p[1::3], p[2::3]
    c = np.empty(4 * N + PAIR_COUNT)
    idx = 0
    
    # Boundary constraints
    c[idx:idx+N] = x - r; idx += N
    c[idx:idx+N] = 1.0 - x - r; idx += N
    c[idx:idx+N] = y - r; idx += N
    c[idx:idx+N] = 1.0 - y - r; idx += N
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dr = r[i_idx] + r[j_idx]
    c[idx:] = dx*dx + dy*dy - dr*dr
    return c

def repair(centers, radii):
    """Iteratively shrinks radii to resolve overlaps and clamp to boundaries."""
    for _ in range(30):
        changed = False
        # Clamp to boundaries
        for i in range(N):
            max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                        centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > max_r - 1e-9:
                radii[i] = max_r
                changed = True
        # Resolve overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-9:
                    ov = radii[i] + radii[j] - d
                    radii[i] -= ov * 0.5
                    radii[j] -= ov * 0.5
                    changed = True
        if not changed:
            break
    return radii

def generate_hex_init(seed, noise_scale=0.01):
    """Generates an initial hexagonal packing configuration with optional noise."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    idx = 0
    y = 0.09
    row = 0
    r = 0.09
    
    while idx < N:
        x_start = r if row % 2 == 0 else 2 * r
        x = x_start
        while x + r <= 1.0 + 1e-9 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r
        y += np.sqrt(3) * r
        row += 1
        
    centers += np.random.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    p = np.zeros(N * 3)
    p[0::3] = centers[:, 0]
    p[1::3] = centers[:, 1]
    p[2::3] = 0.09
    return p

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_p = None
    best_sum = -np.inf
    
    # Phase 1: Diverse Multi-Start Optimization
    for seed in range(30):
        noise = 0.005 * ((seed % 6) + 1)
        p0 = generate_hex_init(seed, noise_scale=noise)
        
        try:
            res = opt.minimize(objective, p0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
            c_vals = constraints(res.x)
            if np.all(c_vals >= -1e-8):
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_p = res.x.copy()
        except Exception:
            continue

    if best_p is None:
        best_p = generate_hex_init(0, noise_scale=0.01)
        
    # Phase 2: Homotopy Growth to escape local minima
    # Gradually expand radii and re-optimize to push density limits
    for step in range(10):
        if best_p is None: break
        
        curr = best_p.copy()
        curr[2::3] *= 1.0035
        curr[0::3] += np.random.normal(0, 0.0008, N)
        curr[1::3] += np.random.normal(0, 0.0008, N)
        curr[0::3] = np.clip(curr[0::3], 0.02, 0.98)
        curr[1::3] = np.clip(curr[1::3], 0.02, 0.98)
        
        try:
            res = opt.minimize(objective, curr, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
            c_vals = constraints(res.x)
            if np.all(c_vals >= -1e-8):
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_p = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final Precise Refinement
    if best_p is not None:
        try:
            res = opt.minimize(objective, best_p, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14})
            if -res.fun > best_sum:
                best_p = res.x.copy()
                best_sum = -res.fun
        except Exception:
            pass

    # Extract and repair
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3].copy()
    radii = repair(centers, radii)
    
    # Final safety check and shrink if necessary to strictly satisfy validation
    for _ in range(10):
        valid = True
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                valid = False
                break
        if not valid: break
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid: break
        if valid: break
        radii *= 0.99995
        
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    return centers, radii, final_sum

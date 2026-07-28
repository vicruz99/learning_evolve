# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=9a47e8a7 sum of radii=2.618175 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Compute all boundary and non-overlap constraints."""
    n = 26
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    c = []
    # Boundary constraints: circle inside [0,1]x[0,1]
    c.append(xs - rs)          # x - r >= 0
    c.append(1.0 - xs - rs)    # 1 - x - r >= 0
    c.append(ys - rs)          # y - r >= 0
    c.append(1.0 - ys - rs)    # 1 - y - r >= 0
    
    # Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    r_sums = rs[:, None] + rs[None, :]
    
    iu, ju = np.triu_indices(n, k=1)
    c.append(dists[iu, ju] - r_sums[iu, ju])
    
    return np.concatenate(c)

def generate_hex_config(pattern, r_target=0.095):
    """Generate a hexagonal lattice configuration based on row counts."""
    n = 26
    pts = []
    y = r_target
    row_idx = 0
    for count in pattern:
        shift = 0.0
        if row_idx % 2 == 1:
            shift = r_target  # Stagger odd rows
            
        if count == 1:
            pts.append([0.5, y])
        else:
            x = r_target + shift
            placed = 0
            while placed < count and x + r_target <= 1.0:
                pts.append([x, y])
                placed += 1
                x += 2 * r_target
                
        y += r_target * 1.7320508  # sqrt(3)
        row_idx += 1
        if len(pts) >= n:
            break
            
    # Fill remaining spots if any
    while len(pts) < n:
        pts.append([0.5, 0.5])
        
    return np.array(pts[:n])

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for [x, y, r] for each of the 26 circles
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # 1. Generate diverse starting configurations
    configs = []
    hex_patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4],
        [5, 6, 6, 5, 4],
        [6, 6, 6, 5, 3],
        [5, 5, 5, 6, 5]
    ]
    
    for p in hex_patterns:
        configs.append(generate_hex_config(p, r_target=0.095))
        
    # Add randomized configurations
    np.random.seed(42)
    for _ in range(15):
        cfg = np.random.rand(n, 2)
        cfg = cfg * 0.8 + 0.1  # Keep centers safely inside initially
        configs.append(cfg)
        
    # 2. Optimization loop
    for cfg in configs:
        x0 = np.zeros(3 * n)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = 0.085  # Start with a feasible radius
        
        # Inject small noise to break symmetry and avoid flat gradients
        x0[:2*n] += np.random.normal(0, 0.005, 2*n)
        
        try:
            res = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 5000, 'ftol': 1e-13}
            )
            
            if res.success:
                rs = res.x[2::3]
                s = np.sum(rs)
                if s > best_sum:
                    best_sum = s
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                    best_radii = rs.copy()
        except Exception:
            continue
            
    # 3. Fallback if optimization yields nothing
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        k = 0
        for i in range(6):
            for j in range(5):
                if k >= n: break
                best_centers[k] = [0.1 + j*0.18, 0.1 + i*0.18]
                best_radii[k] = 0.08
                k += 1
        best_sum = np.sum(best_radii)
        
    # 4. Strict validity repair
    xs = best_centers[:, 0]
    ys = best_centers[:, 1]
    rs = best_radii
    
    # Precompute distance matrix for repair
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    iu, ju = np.triu_indices(n, k=1)
    
    # Check initial validity
    valid = True
    if np.any(rs < 0) or np.any(xs - rs < -1e-9) or np.any(xs + rs > 1 + 1e-9) or \
       np.any(ys - rs < -1e-9) or np.any(ys + rs > 1 + 1e-9):
        valid = False
    if valid:
        r_sums = rs[:, None] + rs[None, :]
        if np.any(dists[iu, ju] < r_sums[iu, ju] - 1e-9):
            valid = False
            
    if not valid:
        # Binary search for the largest scaling factor that makes the packing valid
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            srs = rs * mid
            v = True
            # Check boundaries
            if np.any(xs - srs < -1e-12) or np.any(xs + srs > 1 + 1e-12) or \
               np.any(ys - srs < -1e-12) or np.any(ys + srs > 1 + 1e-12):
                v = False
            # Check overlaps
            if v:
                curr_r_sums = srs[:, None] + srs[None, :]
                if np.any(dists[iu, ju] < curr_r_sums[iu, ju] - 1e-12):
                    v = False
            if v:
                lo = mid
            else:
                hi = mid
        best_radii = rs * lo
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum

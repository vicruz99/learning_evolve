# sol_000075 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 81a0d5f4) state=5b5bfa68 sum of radii=2.621593 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars[2 * N_CIRCLES:])

def constraints(vars):
    """
    Computes inequality constraints >= 0 for valid packing.
    Constraints:
    1. Boundary: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    2. Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    n = N_CIRCLES
    cx = vars[:n]
    cy = vars[n:2 * n]
    r = vars[2 * n:]
    
    c = []
    # Boundary constraints
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise non-overlap constraints (vectorized)
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    
    d2 = cx_m**2 + cy_m**2
    rs2 = r_m**2
    
    # Upper triangular mask to avoid duplicates and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.extend((d2 - rs2)[mask])
    
    return np.array(c)

def get_hex_init(r0):
    """Generates an initial hexagonal grid of N_CIRCLES circles with radius r0."""
    n = N_CIRCLES
    pts = []
    y = r0
    row = 0
    dy = np.sqrt(3) * r0
    while len(pts) < n and y + r0 < 1.0:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2 * r0
        y += dy
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Bounds: centers in [0, 1], radii in [small, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    np.random.seed(42)
    starts = []
    
    # 1. Base hexagonal configuration
    starts.append(get_hex_init(0.10))
    
    # 2. Perturbed hexagonal configurations to escape local minima
    for i in range(15):
        base = get_hex_init(0.10).copy()
        base += np.random.uniform(-0.025, 0.025, base.shape)
        base = np.clip(base, 0.05, 0.95)
        starts.append(base)
        
    # 3. Scaled hex to fill space more aggressively
    base = get_hex_init(0.10).copy()
    mn = base.min(axis=0)
    mx = base.max(axis=0)
    base = (base - mn) / (mx - mn) * 0.8 + 0.1
    starts.append(base)
    starts.append(base + np.random.uniform(-0.015, 0.015, base.shape))
    
    # 4. Random valid starts
    for _ in range(5):
        starts.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    # Prepare upper triangle mask for fast validity checking
    overlap_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    for cfg in starts:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.099  # Strictly feasible initial radius
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': constraints}, 
                options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False}
            )
            
            cx = res.x[:n]
            cy = res.x[n:2 * n]
            r = np.maximum(res.x[2 * n:], 1e-9)
            
            centers = np.column_stack((cx, cy))
            radii = r
            
            # Fast vectorized validity check
            valid = True
            
            # Boundary check
            if np.any(centers[:, 0] < radii - 1e-12) or np.any(centers[:, 0] > 1 - radii + 1e-12) or \
               np.any(centers[:, 1] < radii - 1e-12) or np.any(centers[:, 1] > 1 - radii + 1e-12):
                valid = False
                
            if valid:
                # Overlap check
                cx_m = centers[:, 0][:, None] - centers[:, 0][None, :]
                cy_m = centers[:, 1][:, None] - centers[:, 1][None, :]
                d2 = cx_m**2 + cy_m**2
                rs = radii[:, None] + radii[None, :]
                
                if np.any(d2[overlap_mask] < rs[overlap_mask]**2 - 1e-11):
                    valid = False
                    
            if valid:
                s = np.sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
        except Exception:
            continue

    # Fallback configuration if optimization fails unexpectedly
    if best_centers is None:
        r_fb = 0.095
        pts = []
        y = r_fb
        row = 0
        while len(pts) < n:
            shift = r_fb if row % 2 else 0
            x = r_fb + shift
            while x + r_fb <= 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2 * r_fb
            y += np.sqrt(3) * r_fb
            row += 1
        best_centers = np.array(pts[:n])
        best_radii = np.full(n, r_fb)
        best_sum = np.sum(best_radii)

    # Final safety scaling to guarantee strict validity for the checker's tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)

# sol_000098 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000075 (state 5b5bfa68) state=0cbd0940 sum of radii=2.610558 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)  # Maximize sum(r) -> minimize -sum(r)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        # Radius cannot exceed distance to any boundary
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(max_r, 0.0)))
        
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], 
                            centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def get_hex_init(n, r0=0.10):
    """Generates an initial hexagonal grid of n circles with radius r0."""
    pts = []
    y = r0
    row = 0
    dy = np.sqrt(3) * r0
    while len(pts) < n and y + r0 < 1.0:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += dy
        row += 1
    # Fallback fill if needed
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def objective(vars_array):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars_array[2 * 26:])

def constraints(vars_array):
    """Computes inequality constraints >= 0 for valid packing."""
    n = 26
    cx = vars_array[:n]
    cy = vars_array[n:2 * n]
    r = vars_array[2 * n:]
    
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
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.extend((d2 - rs2)[mask])
    
    return np.array(c)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Bounds: centers in [0, 1], radii in [1e-6, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    np.random.seed(42)
    starts = []
    
    # 1. Base hexagonal configuration
    starts.append(get_hex_init(n, 0.10))
    
    # 2. Perturbed hexagonal configurations to escape local minima
    for i in range(12):
        base = get_hex_init(n, 0.10).copy()
        base += np.random.uniform(-0.03, 0.03, base.shape)
        base = np.clip(base, 0.05, 0.95)
        starts.append(base)
        
    # 3. Scaled hex to fill space more aggressively
    base = get_hex_init(n, 0.10).copy()
    mn = base.min(axis=0)
    mx = base.max(axis=0)
    base = (base - mn) / (mx - mn) * 0.85 + 0.075
    starts.append(base)
    starts.append(base + np.random.uniform(-0.02, 0.02, base.shape))
    
    # 4. Random valid starts
    for _ in range(4):
        starts.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    # Prepare mask for fast validity checking
    overlap_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    for cfg in starts:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.095  # Strictly feasible initial radius
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': constraints}, 
                options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False}
            )
            
            cx = res.x[:n]
            cy = res.x[n:2 * n]
            r = np.maximum(res.x[2 * n:], 1e-9)
            
            centers = np.column_stack((cx, cy))
            radii = r
            
            # Fast vectorized validity check
            valid = True
            
            # Boundary check
            if (np.any(centers[:, 0] < radii - 1e-12) or np.any(centers[:, 0] > 1 - radii + 1e-12) or 
                np.any(centers[:, 1] < radii - 1e-12) or np.any(centers[:, 1] > 1 - radii + 1e-12)):
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

    # Stage 2: Exact LP refinement for radii given the best centers
    # This extracts any remaining slack from the geometric configuration
    lp_radii, lp_sum = solve_lp_radii(best_centers)
    if lp_radii is not None and lp_sum > best_sum:
        best_radii = lp_radii
        best_sum = lp_sum
        
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
                
    best_radii *= max(scale * 0.999999, 0.0)
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)

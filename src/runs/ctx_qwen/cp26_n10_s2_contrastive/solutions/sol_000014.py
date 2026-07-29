# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abe07e0) state=d34ac82b sum of radii=2.591220 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    def objective(x):
        # x contains [x1, y1, r1, x2, y2, r2, ...]
        radii = x[2::3]
        return -np.sum(radii)  # Minimize negative sum to maximize sum

    def boundary_constraints(x):
        # Constraints: r >= 0, x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        c = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            c.append(ri)               # r >= 0
            c.append(xi - ri)          # x - r >= 0
            c.append(1 - (xi + ri))    # x + r <= 1
            c.append(yi - ri)          # y - r >= 0
            c.append(1 - (yi + ri))    # y + r <= 1
        return c

    def no_overlap_constraints(x):
        # Constraints: dist_ij >= ri + rj
        c = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                c.append(np.sqrt(dist_sq) - (ri + rj))
        return c

    # Combine constraints
    def all_constraints(x):
        return boundary_constraints(x) + no_overlap_constraints(x)

    def create_hex_initial(r_scale=0.09):
        # Create a hexagonal lattice initialization
        centers = []
        r = r_scale
        
        # Hexagonal packing parameters
        row_spacing = r * np.sqrt(3)
        col_spacing = 2 * r
        
        y = r
        col_idx = 0
        
        while len(centers) < n:
            x = r if col_idx % 2 == 0 else r + r # Shift every other row
            while x <= 1 - r and len(centers) < n:
                centers.append((x, y, r))
                x += col_spacing
            y += row_spacing
            col_idx += 1
            
        return np.array(centers).flatten()

    best_sol = None
    best_val = -float('inf')

    # Try multiple random restarts to find the global maximum
    for _ in range(5):
        try:
            # Initialize with a slightly smaller hexagonal pattern
            x0 = create_hex_initial(0.08)
            
            # Bounds for x, y, r
            bounds = [(0, 1), (0, 1), (0, 0.5)] * n
            
            # Define constraint structure
            n_constraints = len(all_constraints(x0))
            cons = {'type': 'ineq', 'fun': all_constraints}
            
            # Optimization
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_sol = res.x
                
        except Exception:
            continue

    if best_sol is not None:
        centers = best_sol.reshape(-1, 3)[:, :2]
        radii = best_sol.reshape(-1, 3)[:, 2]
        return centers, radii, float(np.sum(radii))
    
    # Fallback to a simple grid if optimization fails
    fallback_r = 0.1
    centers = []
    radii = []
    x, y = fallback_r, fallback_r
    while len(centers) < n and x + fallback_r <= 1 + 1e-12:
        if y + fallback_r > 1 + 1e-12:
            x += 2 * fallback_r
            y = fallback_r
            continue
        centers.append((x, y))
        radii.append(fallback_r)
        x += 2 * fallback_r
        if x + fallback_r > 1 + 1e-12:
            x = fallback_r
            y += 2 * fallback_r
    
    # Fill remaining if needed with tiny circles
    while len(centers) < n:
        cx, cy = random.random(), random.random()
        centers.append((cx, cy))
        radii.append(1e-6)
        
    return np.array(centers), np.array(radii), float(np.sum(radii))

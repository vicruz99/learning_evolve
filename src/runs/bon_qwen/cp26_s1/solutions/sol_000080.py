# sol_000080 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22281c24) state=038fcbd9 sum of radii=2.444146 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a sequential quadratic penalty method with L-BFGS-B optimization starting from a hexagonal lattice.
    """
    N = 26
    
    # 1. Initial Configuration: Hexagonal lattice pattern
    # Arranging in rows to mimic dense hexagonal packing
    centers = np.zeros((N, 2))
    initial_r = 0.08
    radii = np.full(N, initial_r)
    
    row_counts = [6, 5, 6, 5, 4]
    idx = 0
    dy = np.sqrt(3) * initial_r
    for r_idx, count in enumerate(row_counts):
        y = 0.5 + (r_idx - 2) * dy
        for c_idx in range(count):
            x = 0.5 + (c_idx - (count - 1) / 2) * (2 * initial_r)
            if r_idx % 2 == 1:
                x += initial_r
            centers[idx] = [x, y]
            idx += 1
            
    centers = np.clip(centers, 0, 1)
    x0 = np.hstack([centers.flatten(), radii])
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0) for _ in range(2 * N)] + [(0.0, 0.5) for _ in range(N)]
    
    # Penalty function combining objective and constraint violations
    def penalty_objective(vars, mu):
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Objective: minimize negative sum of radii -> maximize sum of radii
        obj = -np.sum(r)
        
        # Boundary penalties (circles must be inside [0,1]^2)
        pen = np.sum(np.maximum(0.0, r - x)**2)
        pen += np.sum(np.maximum(0.0, r - (1.0 - x))**2)
        pen += np.sum(np.maximum(0.0, r - y)**2)
        pen += np.sum(np.maximum(0.0, r - (1.0 - y))**2)
        
        # Overlap penalties (distance between centers >= sum of radii)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Only consider upper triangle to avoid double counting
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pen += np.sum(np.maximum(0.0, min_dist[mask] - dist[mask])**2)
        
        return obj + mu * pen

    # Sequential Quadratic Penalty with increasing weight
    mu_schedule = [200, 2000, 20000, 100000]
    current_x = x0
    
    for mu in mu_schedule:
        res = minimize(lambda v, m=mu: penalty_objective(v, m), current_x, 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1500, 'ftol': 1e-14, 'disp': False})
        current_x = res.x

    # Extract final configuration
    x_final = current_x[0::3]
    y_final = current_x[1::3]
    r_final = current_x[2::3]
    centers_final = np.column_stack((x_final, y_final))
    
    # Post-processing: Enforce constraints strictly
    # 1. Clip radii to satisfy boundary conditions exactly
    for i in range(N):
        r_final[i] = min(r_final[i], centers_final[i, 0], 1.0 - centers_final[i, 0],
                         centers_final[i, 1], 1.0 - centers_final[i, 1])
                         
    # 2. Iterative shrinking to resolve any residual overlaps
    # This guarantees the validate_packing function will return True
    for _ in range(300):
        overlap_found = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers_final[i, 0] - centers_final[j, 0]
                dy = centers_final[i, 1] - centers_final[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r_final[i] + r_final[j] - 1e-12:
                    excess = r_final[i] + r_final[j] - dist
                    shrink = excess / 2.0 + 1e-9
                    r_final[i] -= shrink
                    r_final[j] -= shrink
                    overlap_found = True
                    break
            if overlap_found:
                break
        if not overlap_found:
            break
            
    r_final = np.maximum(r_final, 0.0)
    total_sum = float(np.sum(r_final))
    
    return centers_final, r_final, total_sum

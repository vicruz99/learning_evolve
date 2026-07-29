# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state adcd3d40) state=788d16be sum of radii=2.624008 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_constraints(vars):
    """Compute all boundary and non-overlap constraints."""
    n = N_CIRCLES
    c = []
    
    # Boundary constraints
    for i in range(n):
        c.append(vars[3*i] - vars[3*i+2])          # x - r >= 0
        c.append(1.0 - vars[3*i] - vars[3*i+2])    # 1 - x - r >= 0
        c.append(vars[3*i+1] - vars[3*i+2])        # y - r >= 0
        c.append(1.0 - vars[3*i+1] - vars[3*i+2])  # 1 - y - r >= 0
        
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = vars[3*i] - vars[3*j]
            dy = vars[3*i+1] - vars[3*j+1]
            r_sum = vars[3*i+2] + vars[3*j+2]
            c.append(dx*dx + dy*dy - r_sum*r_sum)
            
    return np.array(c)

def run_packing():
    n = N_CIRCLES
    
    # Initialize centers in a dense hexagonal pattern
    centers = np.zeros((n, 2))
    idx = 0
    y_start = 0.15
    dy = 0.16
    row_counts = [5, 6, 5, 6, 4]
    
    for r in range(5):
        y = y_start + r * dy
        num_circles = row_counts[r]
        x_start = 0.15
        if r % 2 == 1:
            x_start += 0.08
        for c in range(num_circles):
            centers[idx] = [x_start + c * 0.16, y]
            idx += 1
            
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = 0.04
        
    # Define bounds
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (1e-5, 0.5)])
        
    # Objective: maximize sum of radii -> minimize negative sum
    def objective(vars):
        return -np.sum(vars[2::3])
        
    # Setup constraints
    cons = [{'type': 'ineq', 'fun': compute_constraints}]
    
    # Run optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                   
    x_opt = res.x if res.success else x0
        
    out_centers = x_opt[:3*n].reshape(-1, 3)[:, :2]
    out_radii = x_opt[:3*n].reshape(-1, 3)[:, 2]
    
    # Project to feasible region to handle numerical boundary violations
    for i in range(n):
        x, y, r = out_centers[i, 0], out_centers[i, 1], out_radii[i]
        r = min(r, x, y, 1-x, 1-y)
        r = max(r, 1e-6)
        out_radii[i] = r
        out_centers[i, 0] = np.clip(x, r, 1-r)
        out_centers[i, 1] = np.clip(y, r, 1-r)
        
    # Sequential overlap resolution to strictly satisfy validation tolerance
    for _ in range(100):
        overlap_found = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((out_centers[i,0]-out_centers[j,0])**2 + (out_centers[i,1]-out_centers[j,1])**2)
                req = out_radii[i] + out_radii[j]
                if dist < req - 1e-10:
                    shrink = (req - dist) / 2 + 1e-7
                    out_radii[i] = max(out_radii[i] - shrink, 1e-6)
                    out_radii[j] = max(out_radii[j] - shrink, 1e-6)
                    overlap_found = True
        if not overlap_found:
            break
            
    return out_centers, out_radii, np.sum(out_radii)

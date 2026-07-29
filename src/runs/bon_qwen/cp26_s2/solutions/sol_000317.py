# sol_000317 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1cbfbe8a) state=3eea4663 sum of radii=0.171573 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Returns centers, radii, sum_radii for a packing of 26 circles in a unit square.
    Strategy: Optimize positions and radii using SLSQP starting from a hexagonal grid.
    """
    n = 26
    
    # Objective function: Maximize sum of radii (minimize negative sum)
    # Variables vector: [x1, y1, r1, x2, y2, r2, ...]
    def objective(vars_):
        radii = vars_[2::3]
        return -np.sum(radii)

    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Constraints
    constraints = []
    
    # 1. Boundary constraints: circle inside square
    # x >= r  => x - r >= 0
    # x <= 1-r => 1 - x - r >= 0
    # y >= r  => y - r >= 0
    # y <= 1-r => 1 - y - r >= 0
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[ix] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[ix] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[iy] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[iy] - v[ir]})

    # 2. Non-overlap constraints: distance >= sum of radii
    # Using squared distance: (dx^2 + dy^2) - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            jx, jy, jr = 3*j, 3*j+1, 3*j+2
            
            def dist_sq_constraint(v, i=i, j=j):
                dx = v[ix] - v[jx]
                dy = v[iy] - v[jy]
                ri = v[ir]
                rj = v[jr]
                return (dx*dx + dy*dy) - (ri + rj)**2
            
            constraints.append({'type': 'ineq', 'fun': dist_sq_constraint})

    # Initial guess: Grid pattern dense enough to fit 26 circles
    # 5 columns, 6 rows = 30 points. We use the first 26.
    # Spacing is chosen to be feasible with a small radius.
    x_coords = np.linspace(0.12, 0.88, 5)
    y_coords = np.linspace(0.12, 0.88, 6)
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Select first 26 points
    init_centers = points[:26]
    init_radii = np.full(26, 0.06) # Safe initial radius
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = init_centers[i, 0]
        x0[3*i+1] = init_centers[i, 1]
        x0[3*i+2] = init_radii[i]
        
    best_result = None
    best_sum = -np.inf
    
    # Run optimization with perturbations to escape local minima
    np.random.seed(42)
    for k in range(5):
        x0_perturbed = x0.copy()
        # Add small random noise to positions
        noise = np.random.uniform(-0.02, 0.02, 2 * n)
        x0_perturbed[::3] += noise[:n]
        x0_perturbed[1::3] += noise[n:]
        
        # Clamp positions to [0, 1]
        x0_perturbed[::3] = np.clip(x0_perturbed[::3], 0, 1)
        x0_perturbed[1::3] = np.clip(x0_perturbed[1::3], 0, 1)
        
        try:
            res = opt.minimize(objective, x0_perturbed, method='SLSQP', 
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 2000, 'ftol': 1e-12})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res
        except Exception:
            continue
            
    # Fallback if no successful result found
    if best_result is None:
        try:
            best_result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        except:
            # Return initial valid configuration
            return init_centers, init_radii, np.sum(init_radii)

    # Extract solution
    final_vars = best_result.x
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = final_vars[3*i]
        centers[i, 1] = final_vars[3*i+1]
        radii[i] = final_vars[3*i+2]
        
    # Post-processing to ensure strict validity
    radii = np.maximum(radii, 0.0)
    
    # Enforce boundary constraints strictly
    for i in range(n):
        r = radii[i]
        cx, cy = centers[i]
        cx = np.clip(cx, r, 1.0 - r)
        cy = np.clip(cy, r, 1.0 - r)
        centers[i] = [cx, cy]
        
    # Resolve overlaps by reducing radii if necessary
    for _ in range(50):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                # Allow small tolerance for numerical error, but fix if significant
                if dist < sum_r - 1e-10:
                    scale = dist / sum_r
                    radii[i] *= scale
                    radii[j] *= scale
                    overlap_found = True
        if not overlap_found:
            break
            
    # Re-clip boundaries after radius reduction
    for i in range(n):
        r = radii[i]
        cx, cy = centers[i]
        cx = np.clip(cx, r, 1.0 - r)
        cy = np.clip(cy, r, 1.0 - r)
        centers[i] = [cx, cy]

    final_sum = np.sum(radii)
    return centers, radii, final_sum

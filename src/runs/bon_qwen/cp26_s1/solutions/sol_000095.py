# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5526c41b) state=2e2212e1 sum of radii=2.615828 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    
    # --- Step 1: Initialize with a Hexagonal Lattice ---
    # Estimate radius based on area density heuristic
    # Area = 1. Target density ~0.9. Sum area = 26 * pi * r^2.
    # 26 * pi * r^2 ~ 0.9 -> r ~ 0.105.
    # Start with a slightly smaller radius to ensure validity.
    initial_r = 0.09
    
    centers = np.zeros((n, 2))
    
    # Generate hexagonal packing coordinates
    # Vertical spacing = r * sqrt(3)
    # Horizontal spacing = 2 * r
    y_step = initial_r * np.sqrt(3)
    x_step = 2 * initial_r
    
    idx = 0
    row = 0
    while idx < n:
        # Determine number of circles in this row
        # Offset odd rows by r
        is_odd_row = (row % 2 == 1)
        x_start = initial_r if not is_odd_row else initial_r * 2
        
        # Estimate max circles in row
        # Width available = 1 - 2*r. Step = 2*r.
        # Count = floor((1 - 2*r) / 2*r) + 1 = floor(1/2r - 1) + 1
        count_in_row = int(np.floor((1.0 - 2.0 * initial_r) / (2.0 * initial_r))) + 1
        
        # Ensure we don't exceed total n
        actual_count = min(count_in_row, n - idx)
        
        for i in range(actual_count):
            if idx < n:
                # X coordinate
                if is_odd_row:
                    cx = initial_r + i * x_step
                else:
                    cx = initial_r + i * x_step
                
                # Y coordinate
                cy = initial_r + row * y_step
                
                centers[idx] = [cx, cy]
                idx += 1
        
        row += 1

    # --- Step 2: Force-Directed Relaxation (Pre-optimization) ---
    # This helps resolve overlaps and move circles into valid positions
    # before the strict optimizer runs.
    radii = np.full(n, initial_r)
    
    # Simple repulsive force iteration
    for _ in range(500): # 500 iterations
        # Increase radii slightly to pack tighter
        radii *= 1.001
        
        # Calculate forces
        forces = np.zeros_like(centers)
        
        for i in range(n):
            # Boundary forces (push inside)
            # Left wall
            if centers[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - centers[i, 0]) * 10
            # Right wall
            elif centers[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (centers[i, 0] - (1.0 - radii[i])) * 10
            # Bottom wall
            if centers[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - centers[i, 1]) * 10
            # Top wall
            elif centers[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (centers[i, 1] - (1.0 - radii[i])) * 10
            
            # Inter-circle repulsion
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap * 5.0
                    fy = (dy / dist) * overlap * 5.0
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        # Update positions
        centers += forces * 0.01 # Learning rate
        
        # Clip to valid range [0, 1] to prevent escaping
        centers = np.clip(centers, 1e-9, 1.0 - 1e-9)

    # --- Step 3: Numerical Optimization ---
    # Use scipy.optimize to maximize sum of radii
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    # Total 78 variables.
    
    def objective(vars_flat):
        # Negative sum of radii for minimization
        return -np.sum(vars_flat[2::3])

    def constraint_boundary(vars_flat):
        # Returns array of constraint values (must be >= 0)
        constraints = []
        for i in range(n):
            x = vars_flat[3*i]
            y = vars_flat[3*i + 1]
            r = vars_flat[3*i + 2]
            # x >= r  => x - r >= 0
            constraints.append(x - r)
            # x <= 1-r => 1 - x - r >= 0
            constraints.append(1.0 - x - r)
            # y >= r  => y - r >= 0
            constraints.append(y - r)
            # y <= 1-r => 1 - y - r >= 0
            constraints.append(1.0 - y - r)
            # r >= 0
            constraints.append(r)
        return np.array(constraints)

    def constraint_overlap(vars_flat):
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        constraints = []
        for i in range(n):
            xi, yi, ri = vars_flat[3*i], vars_flat[3*i+1], vars_flat[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = vars_flat[3*j], vars_flat[3*j+1], vars_flat[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist_sq = (ri + rj)**2
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)

    # Prepare initial guess from relaxation
    vars0 = np.zeros(3 * n)
    for i in range(n):
        vars0[3*i] = centers[i, 0]
        vars0[3*i + 1] = centers[i, 1]
        vars0[3*i + 2] = radii[i]

    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    # Bounds for variables (basic bounds, constraints handle tighter ones)
    bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n 

    # Run optimizer
    # SLSQP is suitable for constrained non-linear optimization
    result = minimize(objective, vars0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 1000, 'ftol': 1e-12})

    # Extract results
    final_vars = result.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = final_vars[3*i]
        final_centers[i, 1] = final_vars[3*i + 1]
        final_radii[i] = final_vars[3*i + 2]
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

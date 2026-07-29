# sol_000148 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3a06727e) state=2a1fe2fc sum of radii=2.040149 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # --- Phase 1: Initial Setup and Geometry Optimization ---
    
    # Define number of circles
    n = 26
    
    # Generate initial hexagonal grid positions for 26 circles
    # We use 5 rows with varying number of circles to fill the square well
    centers = []
    row_counts = [6, 5, 6, 5, 4] # Sums to 26
    
    # Calculate row spacing to fit in unit square (y-axis)
    # We leave some margin for the radius optimization
    row_spacing = 1.0 / (len(row_counts) + 1)
    
    y_coords = []
    for i in range(len(row_counts)):
        y_coords.append((i + 1) * row_spacing)
        
    # Generate centers based on hexagonal pattern
    for i, count in enumerate(row_counts):
        # For hexagonal packing, alternate rows are offset
        # Horizontal spacing
        x_spacing = 1.0 / (count + 1)
        y = y_coords[i]
        
        for j in range(count):
            x = (j + 1) * x_spacing
            # Apply hexagonal offset for alternating rows
            if i % 2 == 1:
                x += x_spacing / 2
            
            # Keep within bounds [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Initialize radii to a small value
    radii = np.full(n, 0.01)
    
    # Expand and Relax loop
    # Iteratively increase radii and push circles apart to resolve overlaps
    for _ in range(150):
        # Expand radii
        radii *= 1.005
        
        # Ensure radii are not too large (cap at 0.5 for safety in bounds)
        radii = np.clip(radii, 0, 0.5)
        
        # Define energy function to minimize overlaps
        def energy(c_flat):
            c = c_flat.reshape(n, 2)
            e = 0.0
            r = radii # Fixed during this optimization step
            
            # Pairwise overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        e += overlap**2
            
            # Boundary penalty (soft constraint handled by box, but just in case)
            # Box constraints below handle this strictly, but adding penalty helps stability
            for i in range(n):
                # x constraints
                e += max(0, r[i] - c[i, 0])**2
                e += max(0, c[i, 0] - (1.0 - r[i]))**2
                # y constraints
                e += max(0, r[i] - c[i, 1])**2
                e += max(0, c[i, 1] - (1.0 - r[i]))**2
                
            return e

        # Bounds for centers: [r_i, 1 - r_i] for each coordinate
        bounds = []
        for i in range(n):
            # x bounds
            bounds.append((radii[i], 1.0 - radii[i]))
            # y bounds
            bounds.append((radii[i], 1.0 - radii[i]))
            
        # Run local optimization to resolve overlaps
        res = minimize(energy, centers.flatten(), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-9})
        centers = res.x.reshape(n, 2)

    # --- Phase 2: Final Radial Optimization using Linear Programming ---
    
    # Fix centers and solve for max radii
    # Maximize sum(r_i)
    # Subject to:
    # 1. r_i + r_j <= distance(c_i, c_j)
    # 2. r_i <= x_i, r_i <= 1 - x_i, r_i <= y_i, r_i <= 1 - y_i
    # 3. r_i >= 0
    
    # Calculate pairwise distances
    # We need to construct the constraint matrix A_ub * r <= b_ub
    # Constraints are of the form: r_i + r_j <= d_ij
    
    constraints = []
    rhs = []
    
    # Pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            # Create row for constraint r_i + r_j <= dist
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            constraints.append(row)
            rhs.append(dist)
            
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n); row[i] = 1; constraints.append(row); rhs.append(x)
        # r_i <= 1 - x
        row = np.zeros(n); row[i] = 1; constraints.append(row); rhs.append(1 - x)
        # r_i <= y
        row = np.zeros(n); row[i] = 1; constraints.append(row); rhs.append(y)
        # r_i <= 1 - y
        row = np.zeros(n); row[i] = 1; constraints.append(row); rhs.append(1 - y)
        
    A_ub = np.array(constraints)
    b_ub = np.array(rhs)
    
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    c_obj = np.ones(n) * -1
    
    # Variable bounds: r_i >= 0
    bounds_lp = [(0, None)] * n
    
    # Solve LP
    res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    
    if res_lp.success:
        radii = res_lp.x
    else:
        # Fallback if LP fails, use previous radii (though they should be valid)
        pass
        
    # Calculate final sum of radii
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

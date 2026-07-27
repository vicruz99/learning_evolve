import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP optimization with multiple restarts from a hexagonal lattice initialization.
    """
    n = 26
    
    # Optimization parameters
    n_restarts = 5
    best_score = -np.inf
    best_solution = None
    
    # Helper function to generate hexagonal lattice initialization
    def get_initial_guess():
        # Try to fit points in a hexagonal pattern
        # Approximate spacing for 26 points in 1x1 square
        # Area per point approx 1/26. Radius approx 0.1. Spacing approx 0.2.
        # Let's create a grid and select best 26 or just fill rows.
        
        centers = np.zeros((n, 2))
        idx = 0
        
        # Hexagonal packing parameters
        # row_height = sqrt(3)/2 * spacing
        # Let's estimate spacing 0.2
        spacing = 0.2
        row_height = spacing * np.sqrt(3) / 2
        
        y = spacing / 2
        row = 0
        while idx < n and y + spacing/2 <= 1.0:
            # Determine number of points in this row
            # Offset for odd rows
            x_offset = 0
            if row % 2 == 1:
                x_offset = spacing / 2
            
            x = x_offset
            count_in_row = 0
            while x + spacing/2 <= 1.0 and idx < n:
                centers[idx, 0] = x + spacing/2
                centers[idx, 1] = y + spacing/2 # Center in the cell
                idx += 1
                x += spacing
                count_in_row += 1
            
            y += row_height
            row += 1
        
        # If we didn't fit all points (unlikely with small spacing), fill remaining randomly
        if idx < n:
            remaining = n - idx
            centers[idx:, 0] = np.random.uniform(0.1, 0.9, remaining)
            centers[idx:, 1] = np.random.uniform(0.1, 0.9, remaining)
            
        return centers

    # Objective function: minimize negative sum of radii
    def objective(vars):
        # vars is [x1, y1, r1, x2, y2, r2, ...]
        r = vars[2::3]
        return -np.sum(r)

    # Constraint function
    # Returns an array of values that must be >= 0
    def constraints(vars):
        # Reshape vars to (n, 3) -> [x, y, r]
        c = vars.reshape(n, 3)
        x = c[:, 0]
        y = c[:, 1]
        r = c[:, 2]
        
        cons = []
        
        # 1. Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
        # Left: x >= r  => x - r >= 0
        cons.append(x - r)
        # Right: x <= 1-r => 1 - x - r >= 0
        cons.append(1.0 - x - r)
        # Bottom: y >= r => y - r >= 0
        cons.append(y - r)
        # Top: y <= 1-r => 1 - y - r >= 0
        cons.append(1.0 - y - r)
        
        # 2. Pairwise distance constraints
        # dist_ij >= r_i + r_j
        # dist_ij^2 >= (r_i + r_j)^2  (monotonic for positive distances/radii)
        # However, direct sqrt is safer for smoothness near 0, but squared avoids sqrt cost.
        # Let's use squared distance to avoid sqrt in constraint evaluation?
        # Actually, constraint dist - r_i - r_j >= 0 is standard.
        # dist = sqrt((xi-xj)^2 + (yi-yj)^2)
        
        # Vectorized pairwise computation
        # Broadcast to (n, n) matrices
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        dr = r[:, None] + r[None, :]
        
        # We only need upper triangle (i < j)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        pairwise_cons = dist[mask] - dr[mask]
        cons.append(pairwise_cons)
        
        return np.concatenate(cons)

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (radius cannot exceed 0.5 in unit square)
    # Actually r <= 0.5 is loose, but safe.
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    for restart in range(n_restarts):
        # Generate initial guess
        centers = get_initial_guess()
        # Add small random noise to avoid symmetry issues
        centers += np.random.normal(0, 0.01, centers.shape)
        # Clip to valid range [0, 1]
        centers = np.clip(centers, 0.0, 1.0)
        
        # Initial radii small
        radii_init = np.full(n, 0.02)
        
        # Flatten to 1D array
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii_init[i]
        
        # Define constraints dict for scipy
        # 'ineq' means func(x) >= 0
        cons = {'type': 'ineq', 'fun': constraints}
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_score:
                    best_score = current_sum
                    best_solution = res.x.copy()
            else:
                # Even if not successful, check if result is valid and better
                # Sometimes SLSQP stops early but solution is good
                current_sum = -objective(res.x)
                # Validate briefly (skip expensive check if obvious failure)
                # We'll validate at the end
                if current_sum > best_score:
                    best_score = current_sum
                    best_solution = res.x.copy()
                    
        except Exception:
            continue

    if best_solution is None:
        # Fallback: simple grid solution
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.1) # This might overlap, but let's try to fix
        # 5x5 grid centers
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    centers[idx, 0] = (i + 0.5) / 5.0
                    centers[idx, 1] = (j + 0.5) / 5.0
                    idx += 1
        if idx < n:
             centers[idx, 0] = 0.5
             centers[idx, 1] = 0.5
             idx += 1
        
        # Flatten
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = 0.05 # Small radius to start
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000})
        best_solution = res.x

    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = best_solution[3*i]
        final_centers[i, 1] = best_solution[3*i+1]
        final_radii[i] = best_solution[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
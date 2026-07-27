import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We generate points on a hexagonal grid, which is denser than a square grid.
    # This provides a high-quality starting point for the optimizer.
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05) # Start with a safe, small radius
    
    idx = 0
    row = 0
    y = 0.08 # Initial y position
    
    # Generate hexagonal grid points
    while idx < n:
        is_odd_row = (row % 2 == 1)
        # Hexagonal rows have 5 circles, shifted by half a step
        # Columns are roughly 0.2 apart
        if is_odd_row:
            num_cols = 4
            x_start = 0.1 + 0.1 # Shifted
        else:
            num_cols = 5
            x_start = 0.1
            
        for i in range(num_cols):
            if idx < n:
                centers[idx, 0] = x_start + i * 0.2
                centers[idx, 1] = y
                idx += 1
        
        row += 1
        y += 0.1 * np.sqrt(3) # Vertical spacing for hex packing
        
    # Reshape to optimizer vector format: [x0, y0, r0, x1, y1, r1, ...]
    v0 = np.zeros(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii
    
    # Define bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
    # Objective: Maximize sum of radii (minimize negative sum)
    def objective(v):
        return -np.sum(v[2::3])
    
    # Constraints for SLSQP
    def constraints_func(v):
        cx = v[0::3]
        cy = v[1::3]
        cr = v[2::3]
        
        cons_list = []
        
        # 1. Boundary Constraints
        # Ensure circles stay within [0, 1]
        cons_list.append(cx - cr)          # x - r >= 0
        cons_list.append(1.0 - cx - cr)    # 1 - x - r >= 0
        cons_list.append(cy - cr)          # y - r >= 0
        cons_list.append(1.0 - cy - cr)    # 1 - y - r >= 0
        
        # 2. Pairwise Non-overlap Constraints
        # distance(i, j) >= r_i + r_j
        # Vectorized computation for performance
        C = np.column_stack([cx, cy])
        R = cr
        
        # Compute distance matrix (n, n)
        # diff shape (n, n, 2)
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Compute radius sum matrix (n, n)
        r_sum = R[:, np.newaxis] + R[np.newaxis, :]
        
        # Constraint values: dist - (r_i + r_j)
        # We only need upper triangle (i < j) to avoid duplicates and self-checks
        cons_mat = dists - r_sum
        
        # Extract upper triangular part
        mask = np.triu_indices(n, k=1)
        cons_list.append(cons_mat[mask])
        
        return np.concatenate(cons_list)

    # Run Optimization
    # SLSQP is robust for this type of problem
    try:
        res = minimize(
            objective, 
            v0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints={'type': 'ineq', 'fun': constraints_func},
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        # Extract results
        best_v = res.x
        best_centers = np.zeros((n, 2))
        best_centers[:, 0] = best_v[0::3]
        best_centers[:, 1] = best_v[1::3]
        best_radii = best_v[2::3]
        
    except Exception:
        # Fallback to initial guess if optimization fails
        best_centers = centers
        best_radii = radii
        
    sum_radii = float(np.sum(best_radii))
    
    return (best_centers, best_radii, sum_radii)
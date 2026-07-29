# sol_000205 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c1389c4d) state=38c7fbb5 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Constant for number of circles
N_CIRCLES = 26

def objective_function(x):
    """
    Objective function to maximize sum of radii.
    Minimize negative sum.
    x: array of shape (3*N,) containing [x1, y1, r1, x2, y2, r2, ...]
    """
    # Radii are at indices 2, 5, 8, ... (step 3 starting from 2)
    radii = x[2::3]
    return -np.sum(radii)

def constraints_function(x):
    """
    Computes inequality constraints.
    Returns an array of values, all of which must be >= 0.
    x: array of shape (3*N,)
    """
    n = N_CIRCLES
    
    # Extract coordinates and radii
    # x: [x0, y0, r0, x1, y1, r1, ...]
    x_coords = x[0::3]
    y_coords = x[1::3]
    radii = x[2::3]
    
    # 1. Boundary Constraints
    # For each circle i:
    # x_i >= r_i  => x_i - r_i >= 0
    # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i => y_i - r_i >= 0
    # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
    
    # Preallocate array for boundary constraints
    # 4 constraints per circle
    c_boundary = np.empty(4 * n)
    
    # Fill constraints using slicing
    # Indices for circle i's constraints are 4*i, 4*i+1, 4*i+2, 4*i+3
    # Slice c[0::4] picks indices 0, 4, 8... which correspond to the first constraint of each circle
    c_boundary[0::4] = x_coords - radii
    c_boundary[1::4] = 1.0 - x_coords - radii
    c_boundary[2::4] = y_coords - radii
    c_boundary[3::4] = 1.0 - y_coords - radii
    
    # 2. Overlap Constraints
    # For each pair i < j:
    # dist(i, j)^2 >= (r_i + r_j)^2
    # dist(i, j)^2 - (r_i + r_j)^2 >= 0
    
    centers = np.column_stack((x_coords, y_coords))
    
    # Compute pairwise squared Euclidean distances
    # pdist returns condensed distance matrix (1D array)
    dists = pdist(centers)
    sq_dists = dists**2
    
    # Compute sum of radii for all pairs
    # We need r_i + r_j for i < j
    # Create matrix of sums
    sum_radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract upper triangular part (excluding diagonal) corresponding to pairs i < j
    # The order of elements in pdist matches the row-major traversal of the upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sum_radii_pairs = sum_radii_matrix[mask]
    
    c_overlap = sq_dists - sum_radii_pairs**2
    
    # Combine all constraints
    all_constraints = np.concatenate((c_boundary, c_overlap))
    
    return all_constraints

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float, the sum of all radii
    """
    n = N_CIRCLES
    
    # --- 1. Initialization ---
    # Initialize with a hexagonal lattice pattern inside the unit square.
    # We choose a radius small enough to fit, e.g., 0.09.
    r_init = 0.09
    centers_list = []
    
    row_idx = 0
    count = 0
    
    # Generate rows
    while count < n:
        # Vertical position of the row center
        y = r_init + row_idx * (r_init * np.sqrt(3))
        
        # Horizontal shift for odd rows (hexagonal packing)
        shift = (row_idx % 2) * r_init
        
        # Start x position
        x = r_init + shift
        
        # Place circles in this row
        while True:
            centers_list.append([x, y])
            count += 1
            if count >= n:
                break
            
            # Move to next circle in row
            x += 2 * r_init
            
            # Check if next circle fits in width
            if x + r_init > 1.0 + 1e-9:
                break
        
        row_idx += 1
        
        # Safety break if y goes too high
        if y + r_init > 1.0 + 1e-9 and row_idx > 6:
            break

    # Fill remaining if any (should not happen with r=0.09)
    while len(centers_list) < n:
        centers_list.append([0.5, 0.5])
        
    centers_init = np.array(centers_list[:n])
    radii_init = np.full(n, r_init)
    
    # Flatten variables for optimizer: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    x0[0::3] = centers_init[:, 0]
    x0[1::3] = centers_init[:, 1]
    x0[2::3] = radii_init
    
    # Bounds: x, y in [0, 1], r in [0, 1]
    bounds = [(0.0, 1.0)] * (3 * n)
    
    # Define constraints
    cons = {
        'type': 'ineq',
        'fun': constraints_function
    }
    
    # --- 2. Optimization ---
    # Use SLSQP method for constrained optimization
    try:
        res = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        x_opt = res.x
    except Exception:
        # Fallback to initial solution if optimization fails
        x_opt = x0
    
    # --- 3. Extract Results ---
    final_x = x_opt[0::3]
    final_y = x_opt[1::3]
    final_r = x_opt[2::3]
    
    centers_out = np.column_stack((final_x, final_y))
    radii_out = final_r
    
    # Ensure radii are non-negative
    radii_out = np.maximum(radii_out, 0.0)
    
    sum_r = np.sum(radii_out)
    
    return centers_out, radii_out, sum_r

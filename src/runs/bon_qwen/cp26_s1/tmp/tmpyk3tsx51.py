import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# Global constant for the number of circles
N_CIRCLES = 26

def constraint_func_26(x):
    """
    Computes all constraint violations for the optimization.
    x is a 1D array of shape (3 * N_CIRCLES):
    [x_0, y_0, x_1, y_1, ..., x_n, y_n, r_0, r_1, ..., r_n]
    Returns an array of constraint values where value >= 0 indicates satisfaction.
    """
    # Reshape centers and radii
    centers = x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = x[2 * N_CIRCLES:]
    
    constraints = []
    
    # 1. Boundary Constraints
    # Circle i must be inside [0, 1] x [0, 1]
    # x_i - r_i >= 0
    constraints.append(centers[:, 0] - radii)
    # 1 - x_i - r_i >= 0  => x_i + r_i <= 1
    constraints.append(1.0 - centers[:, 0] - radii)
    # y_i - r_i >= 0
    constraints.append(centers[:, 1] - radii)
    # 1 - y_i - r_i >= 0  => y_i + r_i <= 1
    constraints.append(1.0 - centers[:, 1] - radii)
    # Radii must be positive (lower bound)
    constraints.append(radii - 1e-6)
    
    # 2. Overlap Constraints
    # Distance between centers i and j must be >= r_i + r_j
    # Equivalent to dist^2 >= (r_i + r_j)^2
    
    # Compute pairwise squared distances
    # centers shape: (N, 2)
    # diff shape: (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2) # Shape (N, N)
    
    # Compute pairwise sum of radii squared
    # radii shape: (N,)
    # sum_r shape: (N, N)
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    sum_r_sq = sum_r**2
    
    # Constraint matrix: dist_sq - sum_r_sq >= 0
    overlap_mat = dist_sq - sum_r_sq
    
    # We only need constraints for i < j (upper triangle)
    # Using triu_indices to select elements
    triu_idx = np.triu_indices(N_CIRCLES, k=1)
    constraints.append(overlap_mat[triu_idx])
    
    return np.concatenate(constraints)

def get_initial_packing_hexagonal(n=26, r_init=0.05):
    """
    Generates a valid initial hexagonal packing configuration.
    """
    centers = []
    radii = []
    
    # Distribute n circles into rows
    # Using 6 rows for 26 circles
    num_rows = 6
    base_count = n // num_rows
    extra_circles = n % num_rows
    
    # Assign counts to rows
    row_counts = []
    for i in range(num_rows):
        count = base_count
        if i < extra_circles:
            count += 1
        row_counts.append(count)
    
    # Place circles
    vertical_step = r_init * np.sqrt(3)
    current_y = r_init
    
    for row_idx, count in enumerate(row_counts):
        # Hexagonal shift: even rows start at r, odd rows start at 2r
        if row_idx % 2 == 0:
            start_x = r_init
        else:
            start_x = 2 * r_init
        
        horizontal_step = 2 * r_init
        
        for k in range(count):
            cx = start_x + k * horizontal_step
            cy = current_y
            centers.append([cx, cy])
            radii.append(r_init)
        
        current_y += vertical_step
        
    return np.array(centers), np.array(radii)

def run_packing():
    # 1. Generate initial valid packing
    centers, radii = get_initial_packing_hexagonal(N_CIRCLES, r_init=0.05)
    
    # 2. Define objective function (maximize sum of radii -> minimize negative sum)
    def objective(x):
        radii_curr = x[2 * N_CIRCLES:]
        return -np.sum(radii_curr)
        
    # 3. Prepare optimization variables
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x, y in [0, 1], r in [1e-6, 0.5]
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0, 1))      # x
        bounds.append((0, 1))      # y
        bounds.append((1e-6, 0.5)) # r
        
    # 4. Define constraints
    # Using NonlinearConstraint with the vectorized constraint function
    cons = [NonlinearConstraint(constraint_func_26, 0, np.inf)]
    
    # 5. Run Optimization
    # SLSQP is suitable for non-linear problems with constraints
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
    
    best_x = res.x
    best_centers = best_x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    best_radii = best_x[2 * N_CIRCLES:]
    
    # 6. Post-processing to ensure strict validity
    # Clamp radii to fit within boundaries
    for i in range(N_CIRCLES):
        x, y = best_centers[i]
        r = best_radii[i]
        # Max radius allowed by boundaries
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if max_r < 1e-9:
            max_r = 1e-9
        if r > max_r + 1e-12:
            best_radii[i] = max_r
            
    # Fix overlaps by shrinking radii
    # Iterate to resolve any remaining overlaps
    for _ in range(100):
        overlap_found = False
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = best_radii[i] + best_radii[j]
                
                if dist < r_sum - 1e-10:
                    overlap_found = True
                    # Reduce radii to eliminate overlap
                    # Reduce sum by overlap amount + epsilon
                    reduction = (r_sum - dist) + 1e-10
                    # Distribute reduction equally
                    red = reduction / 2.0
                    best_radii[i] -= red
                    best_radii[j] -= red
                    
        if not overlap_found:
            break
            
    # Final boundary check after radius reduction
    for i in range(N_CIRCLES):
        x, y = best_centers[i]
        r = best_radii[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if max_r < 1e-9:
            max_r = 1e-9
        if r > max_r:
            best_radii[i] = max_r
            
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii
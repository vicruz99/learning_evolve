import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple[np.ndarray, np.ndarray, float]: (centers, radii, sum_radii)
    """
    n_circles = 26
    
    # Helper function to define the objective
    def objective(vars_vec):
        # vars_vec contains [x0, y0, r0, x1, y1, r1, ...]
        radii = vars_vec[2::3]
        return -np.sum(radii) # Minimize negative sum = Maximize sum

    # Helper function to define constraints
    def get_constraints(n, vars_vec):
        # Reshape variables
        centers = np.array([vars_vec[0::3], vars_vec[1::3]]).T
        radii = vars_vec[2::3]
        
        constraints_list = []
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # Equivalent to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        for i in range(n):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})          # x - r >= 0
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})     # 1 - x - r >= 0
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})       # y - r >= 0
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})   # 1 - y - r >= 0
            
        # Overlap constraints: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                def overlap_constraint(v, i=i, j=j):
                    x_i, y_i, r_i = v[3*i], v[3*i+1], v[3*i+2]
                    x_j, y_j, r_j = v[3*j], v[3*j+1], v[3*j+2]
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    sum_r_sq = (r_i + r_j)**2
                    return dist_sq - sum_r_sq
                constraints_list.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return constraints_list

    def optimize_packing(init_centers, init_radii, random_seed=42):
        n = len(init_radii)
        # Combine into a single vector
        x_init = init_centers[:, 0].flatten()
        y_init = init_centers[:, 1].flatten()
        r_init = init_radii.flatten()
        
        # Interleave to match [x0, y0, r0, x1, y1, r1, ...] format for easier indexing in constraints if needed, 
        # but actually let's keep them separate in the array for cleaner logic? 
        # Let's stick to the format: [x0, y0, r0, x1, y1, r1, ...]
        vars_init = np.zeros(3 * n)
        vars_init[0::3] = x_init
        vars_init[1::3] = y_init
        vars_init[2::3] = r_init
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = [(0, 1) for _ in range(3 * n)]
        # Tighten radius bounds slightly to avoid degenerate solutions, though >= 0 is handled
        for i in range(0, 3*n, 3):
            bounds[i+2] = (1e-6, 0.5)

        constraints = get_constraints(n, vars_init)
        
        # Use SLSQP
        result = opt.minimize(
            objective, 
            vars_init, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            final_x = result.x[0::3]
            final_y = result.x[1::3]
            final_r = result.x[2::3]
            final_centers = np.column_stack((final_x, final_y))
            final_sum_r = np.sum(final_r)
            return final_centers, final_r, final_sum_r
        
        return None, None, -1.0

    def generate_hexagonal_init(seed):
        np.random.seed(seed)
        n = 26
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.08) # Start with a reasonable small radius
        
        # Generate a hexagonal-like grid
        # We want to distribute points roughly evenly
        # 5 rows of roughly 5-6 points
        rows = 5
        cols_per_row = [6, 5, 6, 5, 4] # Sum = 26
        
        idx = 0
        for r_idx in range(rows):
            n_cols = cols_per_row[r_idx]
            # y coordinate
            y = 0.1 + r_idx * 0.2 
            # x coordinates
            # Shift every other row for hexagonal pattern
            x_start = 0.1 + (r_idx % 2) * 0.1
            step = (1 - 2*0.1) / (n_cols - 1) if n_cols > 1 else 0
            for c_idx in range(n_cols):
                x = x_start + c_idx * step
                # Add small noise to break symmetry and help optimization
                x += np.random.normal(0, 0.005)
                y += np.random.normal(0, 0.005)
                
                # Clip to valid range
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                
                centers[idx] = [x, y]
                idx += 1
                
        return centers, radii

    best_sum_r = -1.0
    best_centers = None
    best_radii = None

    # Try multiple random seeds to find a good local optimum
    seeds = [42, 123, 456, 789, 1010]
    
    for seed in seeds:
        init_centers, init_radii = generate_hexagonal_init(seed)
        centers, radii, sum_r = optimize_packing(init_centers, init_radii)
        
        if centers is not None and sum_r > best_sum_r:
            best_sum_r = sum_r
            best_centers = centers
            best_radii = radii

    # Validate result before returning (just a sanity check logic, validation function is separate)
    # Ensure non-negative radii
    if best_radii is not None:
        best_radii = np.maximum(best_radii, 1e-9)
        
    return best_centers, best_radii, best_sum_r

# For local testing only
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Best Sum of Radii: {sum_r}")
    # import validation_module
    # print(validation_module.validate_packing(centers, radii))
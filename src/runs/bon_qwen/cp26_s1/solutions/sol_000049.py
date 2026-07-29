# sol_000049 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f1389a1) state=f361a347 sum of radii=2.372827 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def get_max_sum_radii(centers):
    """
    Solves the LP to find the maximum sum of radii for a fixed set of centers.
    """
    n = centers.shape[0]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Constraints matrix A_ub * r <= b_ub
    A_ub = []
    b_ub = []
    
    # 1. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(x)
        
        # r_i <= 1 - x
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1.0 - x)
        
        # r_i <= y
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(y)
        
        # r_i <= 1 - y
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1.0 - y)

    # 2. Pairwise constraints: r_i + r_j <= dist(i, j)
    # To avoid O(N^2) matrix construction overhead in a tight loop, we can use sparse, 
    # but for N=26 dense is fine and simple.
    # However, constructing a large dense matrix every call is costly.
    # We will build it only when needed, but N=26 is small enough.
    
    # Pre-calculate distances
    # dist_matrix[i, j] = distance between center i and center j
    # We only need i < j
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x # Return sum of radii and the radii array
        else:
            # Fallback to a valid but small packing if LP fails (should be rare)
            small_r = 0.01
            return 26 * small_r, np.full(n, small_r)
    except Exception:
        return 0.0, np.zeros(n)

def objective(centers_flat):
    """
    Objective function for scipy.optimize.minimize.
    """
    centers = centers_flat.reshape(26, 2)
    # Ensure centers are within [0,1] to prevent invalid LP bounds (negative distances to walls)
    # Though linprog handles negative b_ub by infeasibility, it's safer to clamp.
    centers = np.clip(centers, 0, 1)
    
    sum_r, radii = get_max_sum_radii(centers)
    return -sum_r # Minimize negative sum

def run_packing():
    # 1. Initialize centers in a perturbed hexagonal grid
    centers = np.zeros((26, 2))
    
    # Approximate layout: 5 rows
    # Pattern of circles per row: 6, 5, 6, 5, 4 (Total 26)
    row_counts = [6, 5, 6, 5, 4]
    y_spacing = 1.0 / 6.0 # Rough vertical spacing
    
    current_idx = 0
    for row_idx, count in enumerate(row_counts):
        y = 0.15 + row_idx * y_spacing * 1.1 # Slightly spread out vertically
        # Shift odd rows (1, 3) for hexagonal packing
        offset = 0.0
        if row_idx % 2 == 1:
            offset = 0.5 / max(1, count-1) if count > 1 else 0
        
        x_start = 0.1
        x_end = 0.9
        if count > 1:
            x_step = (x_end - x_start) / (count - 1)
        else:
            x_step = 0
            
        for k in range(count):
            x = x_start + k * x_step + offset
            centers[current_idx] = [x, y]
            current_idx += 1
            
    # Add some noise to break symmetry and help optimization escape local minima
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)

    # 2. Optimize centers using Nelder-Mead
    # Flatten centers for the optimizer
    x0 = centers.flatten()
    
    # Use Nelder-Mead as it doesn't require gradients and works for non-smooth objectives
    # The objective function involves an LP, which is convex but the mapping from centers to max radii is non-smooth.
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    final_centers = res.x.reshape(26, 2)
    
    # 3. Get final radii using the optimized centers
    # Re-solve LP to get exact radii for the final configuration
    sum_r, final_radii = get_max_sum_radii(final_centers)
    
    # Round small negative radii (if any) to 0
    final_radii = np.maximum(final_radii, 0)
    
    # Recalculate sum to be precise
    final_sum = np.sum(final_radii)
    
    return final_centers, final_radii, final_sum

# Helper to run and print results if executed directly
if __name__ == "__main__":
    # Set seed for reproducibility in thought process, though not required by prompt
    np.random.seed(42)
    centers, radii, total_r = run_packing()
    
    # Validation check
    # (Assuming validate_packing is available in environment)
    # print(validate_packing(centers, radii))
    
    print(f"Sum of radii: {total_r}")
    print(f"Min radius: {np.min(radii)}")
    print(f"Max radius: {np.max(radii)}")
    print(f"Centers shape: {centers.shape}")
    print(f"Radii shape: {radii.shape}")

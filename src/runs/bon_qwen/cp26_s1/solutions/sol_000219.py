# sol_000219 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5b6844e7) state=6b746fee sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(123)  # For reproducibility

    # 1. Initialization: Hexagonal grid
    # We start with a radius slightly smaller than optimal to allow optimization room
    r_init = 0.09
    row_spacing = r_init * math.sqrt(3)
    col_spacing = 2 * r_init
    
    centers = []
    # Try to fit as many as possible in a hex grid within [0,1]x[0,1]
    y = r_init
    while len(centers) < n:
        x = r_init
        row_idx = 0
        # Shift odd rows by r_init
        shift = (int((y - r_init) / row_spacing) % 2) * r_init
        x = r_init + shift
        
        while x <= 1 - r_init + 1e-9 and len(centers) < n:
            centers.append([x, y])
            x += col_spacing
        y += row_spacing
    
    # If we haven't reached 26, fill the rest randomly or just extend the grid logic
    # The above logic with r=0.09 might fit 26.
    # If not, we pad with random points (optimizer will fix them)
    while len(centers) < n:
        centers.append([np.random.rand(), np.random.rand()])
        
    centers = np.array(centers[:n])
    
    # 2. Setup for iterative optimization
    # We will perform hill climbing on centers
    best_sum_radii = -1.0
    best_centers = centers.copy()
    best_radii = np.ones(n) * r_init

    # Pre-allocate pair indices for LP constraints
    # Constraints: r_i + r_j <= d_ij
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    n_pairs = len(pair_indices)

    # Helper to solve for radii given centers
    def get_optimal_radii(centers_arr):
        # 1. Compute boundary limits U_i
        # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        x = centers_arr[:, 0]
        y = centers_arr[:, 1]
        U = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # 2. Compute distances d_ij
        # Using broadcasting for O(N^2)
        diff = centers_arr[:, np.newaxis, :] - centers_arr[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # 3. Setup LP
        # Minimize -sum(r_i) => c = -1
        c_obj = -np.ones(n)
        
        # Inequality constraints: r_i + r_j <= d_ij
        # Matrix A_ub of shape (n_pairs, n)
        # This is sparse, but for N=26 dense is fine
        A_ub = np.zeros((n_pairs, n))
        b_ub = np.zeros(n_pairs)
        
        for k, (i, j) in enumerate(pair_indices):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dists[i, j]
            
        # Bounds: 0 <= r_i <= U_i
        bounds = [(0, u) for u in U]
        
        # Solve
        # method='highs' is fast and robust
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.fun * -1, res.x
        else:
            # Fallback if LP fails (rare)
            return 0.0, np.zeros(n)

    # 3. Optimization Loop
    # We run a simple simulated annealing / hill climbing
    temperature = 0.1
    n_iterations = 2000
    
    current_centers = centers.copy()
    
    for step in range(n_iterations):
        # Get current best sum and radii
        current_sum, current_radii = get_optimal_radii(current_centers)
        
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()
            
            # Increase temperature or keep if we found improvement?
            # Let's just use a simple cooling schedule
            temperature = max(0.01, temperature * 0.999)
            
        # Perturb centers
        # Pick a random circle to move
        idx = np.random.randint(n)
        
        # Move in a random direction
        direction = np.random.randn(2)
        step_size = temperature * 0.1
        new_center = current_centers[idx] + direction * step_size
        
        # Keep within [0, 1]
        new_center = np.clip(new_center, 0, 1)
        
        # Apply perturbation
        current_centers[idx] = new_center
        
        # Occasionally perturb multiple circles for better exploration
        if np.random.rand() < 0.1:
            for _ in range(3):
                idx2 = np.random.randint(n)
                dir2 = np.random.randn(2)
                current_centers[idx2] = np.clip(current_centers[idx2] + dir2 * step_size, 0, 1)

    # 4. Final Validation and Return
    # Ensure the best configuration is valid
    # Run one last check/optimization on best_centers
    final_sum, final_radii = get_optimal_radii(best_centers)
    
    # Double check validity manually to be safe
    # (Though LP guarantees constraints, numerical errors might occur)
    # The LP constraints were r_i + r_j <= d_ij.
    # If d_ij is computed from best_centers, it should hold.
    
    return best_centers, final_radii, float(final_sum)

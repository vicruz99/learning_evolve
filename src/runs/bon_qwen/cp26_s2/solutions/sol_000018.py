# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfb3fe63) state=1a7266f8 sum of radii=2.357951 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generates a valid packing of 26 circles in a unit square maximizing the sum of radii.
    """
    n = 26
    
    # 1. Initialize centers with a hexagonal grid pattern
    centers = []
    # 6 rows with lengths [5, 4, 5, 4, 5, 3] sums to 26
    row_lengths = [5, 4, 5, 4, 5, 3]
    
    # Constants for hexagonal packing
    # We will adjust spacing dynamically in the optimization, 
    # but this gives a good topological start.
    for i, count in enumerate(row_lengths):
        y_base = 0.1 + i * 0.16  # Approximate vertical spacing
        x_start = 0.15 if i % 2 == 0 else 0.15 + 0.16 # Approximate horizontal shift
        
        for j in range(count):
            x = x_start + j * 0.16
            # Clamp to ensure inside [0,1]
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y_base, 0.01, 0.99)
            centers.append([x, y])
            
    centers = np.array(centers)

    def get_optimal_radii(current_centers):
        """
        Solves an LP to find the radii that maximize sum(r) for fixed centers.
        """
        num_circles = len(current_centers)
        c_obj = -np.ones(num_circles) # Maximize sum r => minimize -sum r

        A_ub = []
        b_ub = []
        
        # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
        for i in range(num_circles):
            x, y = current_centers[i]
            for val in [x, 1 - x, y, 1 - y]:
                row = np.zeros(num_circles)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(val)
                
        # Overlap constraints: r_i + r_j <= dist_ij
        # To save time, we can just add all pairs or nearest neighbors.
        # For n=26, 325 pairs is small.
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                row = np.zeros(num_circles)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist)
                
        # Bounds for r_i >= 0
        bounds = [(0, None) for _ in range(num_circles)]
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x
        else:
            # Fallback to small equal radii if LP fails
            return np.full(num_circles, 0.01)

    # 2. Iterative refinement
    best_centers = centers.copy()
    best_radii = get_optimal_radii(best_centers)
    best_sum = np.sum(best_radii)
    
    # Try random perturbations to improve layout
    num_iterations = 2000
    step_size = 0.02
    
    for _ in range(num_iterations):
        # Perturb a random circle
        idx = np.random.randint(0, n)
        noise = np.random.uniform(-step_size, step_size, 2)
        new_centers = best_centers.copy()
        new_centers[idx] += noise
        
        # Clamp to valid range
        new_centers = np.clip(new_centers, 0.001, 0.999)
        
        # Solve LP for new centers
        radii = get_optimal_radii(new_centers)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = new_centers
            best_radii = radii
            
    return best_centers, best_radii, float(best_sum)

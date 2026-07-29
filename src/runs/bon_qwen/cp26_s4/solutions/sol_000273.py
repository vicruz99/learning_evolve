# sol_000273 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c64acbd5) state=9eb43f8e sum of radii=1.972364 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def calculate_optimal_radii(centers):
    """
    Solves the Linear Programming problem to find the maximum sum of radii
    for a fixed set of centers.
    """
    n = centers.shape[0]
    
    # 1. Compute boundary constraints (wall distances)
    x, y = centers[:, 0], centers[:, 1]
    w_i = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # 2. Compute pairwise distance constraints
    pairs = []
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            d_ij = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            dists.append(d_ij)
    
    num_pairs = len(pairs)
    
    # 3. Setup LP
    # Maximize sum(r_i) -> Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints: r_i + r_j <= d_ij
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.array(dists)
    
    for k, (i, j) in enumerate(pairs):
        A_ub[k, i] = 1
        A_ub[k, j] = 1
        
    # Bounds: 0 <= r_i <= w_i
    bounds = [(0, w_i[i]) for i in range(n)]
    
    # 4. Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, -res.fun
    else:
        # Fallback to zero if infeasible (should not happen with 0 lower bound)
        return np.zeros(n), 0.0

def score_function(center_flat):
    """
    Objective function for center optimization.
    Returns negative sum of radii because we are minimizing.
    """
    centers = center_flat.reshape(-1, 2)
    radii, sum_r = calculate_optimal_radii(centers)
    return -sum_r

def run_packing():
    # 1. Initialization: Hexagonal Grid Layout
    # We distribute 26 points into 5 rows with alternating counts to fit the square
    # Row counts: 6, 5, 6, 5, 4 = 26 points
    row_counts = [6, 5, 6, 5, 4]
    centers = []
    
    # Vertical spacing adjusted for hexagonal packing (sqrt(3)/2 factor relative to width)
    # We use a uniform vertical distribution to start, optimization will fix the aspect ratio
    row_height = 1.0 / 5.0 
    
    current_idx = 0
    for row_idx, count in enumerate(row_counts):
        y = (row_idx + 0.5) * row_height
        
        # Horizontal spacing
        x_spacing = 1.0 / (count + 1) # Add 1 to leave margin on both sides
        
        # Shift odd rows (indices 1, 3) to create hexagonal offset
        if row_idx % 2 != 0:
            x_offset = x_spacing / 2
        else:
            x_offset = x_spacing
            
        for col in range(count):
            x = (col + 1) * x_spacing + x_offset
            # Clip to ensure strictly inside [0, 1] before optimization
            centers.append([np.clip(x, 1e-6, 1-1e-6), np.clip(y, 1e-6, 1-1e-6)])
            
    centers = np.array(centers)
    
    # 2. Optimization of Centers
    # Using Nelder-Mead to maximize the sum of radii by finding optimal positions
    initial_flat = centers.flatten()
    
    # Optimization with a moderate number of iterations to find a good local optimum
    result = minimize(score_function, initial_flat, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    optimal_centers = result.x.reshape(-1, 2)
    
    # 3. Final Calculation of Radii
    radii, sum_radii = calculate_optimal_radii(optimal_centers)
    
    return optimal_centers, radii, sum_radii

# Validation helper (for self-check, not part of required interface)
def validate_packing(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 0: return False
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")

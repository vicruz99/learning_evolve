# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f223c9a2) state=ee32a6ee sum of radii=1.925262 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # --- Stage 1: Hexagonal Grid Initialization ---
    # Generate a hexagonal lattice of points
    cols = 6
    rows = 5
    pts = []
    
    # Base spacing estimates
    x_step = 1.0 / (cols - 0.5)
    y_step = x_step * np.sqrt(3) / 2
    
    # Create rows, shifting even rows to form a hexagonal pattern
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5 + (0.5 if r % 2 == 1 else 0)) * x_step
            y = (r + 0.5) * y_step
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
    
    # We might have too many or too few points, so we filter and shuffle to pick 26
    np.random.seed(42) # For reproducibility
    pts = np.array(pts)
    if len(pts) > n:
        indices = np.random.choice(len(pts), n, replace=False)
        centers = pts[indices]
    elif len(pts) < n:
        # Fallback to random fill if grid is too small
        centers = np.random.rand(n, 2)
    else:
        centers = pts

    # --- Stage 2: Equal Radii Expansion (Force Directed) ---
    # We optimize the position to maximize the minimum distance between centers and boundaries.
    # This is equivalent to finding the largest equal radius 'r'.
    
    r = 0.01
    centers = centers * 0.8 + 0.1 # Scale into box
    
    # Function to calculate max equal radius for given centers
    def max_equal_radius(c):
        # Distance to boundaries
        b_min = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), 
                           np.minimum(c[:, 1], 1 - c[:, 1]))
        min_b = np.min(b_min)
        
        # Distance between centers
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                dists.append(d)
        min_d = np.min(dists) if dists else 1.0
        
        return np.min([min_b, min_d / 2.0])

    # Initial radius estimate
    r = max_equal_radius(centers)
    
    # Relaxation loop
    # We try to improve the position by gradient ascent on the bottleneck distance
    # Since it's non-smooth, we use a local search (Nelder-Mead) from the grid start
    def objective_for_optimization(x_flat):
        c = x_flat.reshape(n, 2)
        return -max_equal_radius(c)

    res = minimize(objective_for_optimization, centers.flatten(), method='Nelder-Mead', 
                   options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-5})
    centers = res.x.reshape(n, 2)
    r = -res.fun
    radii[:] = r

    # --- Stage 3: LP Refinement for Non-Equal Radii ---
    # Fix centers, maximize sum of radii using Linear Programming.
    # Constraints:
    # 1. r_i <= x_i <= 1 - r_i  => r_i - x_i <= 0, r_i + x_i <= 1
    # 2. r_i <= y_i <= 1 - r_i  => r_i - y_i <= 0, r_i + y_i <= 1
    # 3. r_i + r_j <= dist(i, j) => r_i + r_j <= dist_ij

    # Variable vector: [r_0, r_1, ..., r_25]
    c_obj = -np.ones(n) # Maximize sum => Minimize negative sum

    # Inequality constraints A_ub @ vars <= b_ub
    # We need to build A_ub and b_ub
    # 4 constraints per circle for boundaries
    # 1 constraint per pair for non-overlap
    
    m = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        xi, yi = centers[i]
        # r_i - x_i <= 0
        A_ub[idx, i] = 1.0
        b_ub[idx] = xi
        idx += 1
        
        # r_i + x_i <= 1
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1 - xi
        idx += 1
        
        # r_i - y_i <= 0
        A_ub[idx, i] = 1.0
        b_ub[idx] = yi
        idx += 1
        
        # r_i + y_i <= 1
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1 - yi
        idx += 1

    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    lp_res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if lp_res.success:
        radii = lp_res.x
    else:
        # Fallback to equal radii if LP fails
        radii[:] = r

    # --- Final Validation and Return ---
    sum_r = np.sum(radii)
    
    # Small sanity check/cleanup (though LP should guarantee validity)
    # Just ensuring no negatives due to numerical noise
    radii = np.maximum(radii, 0)

    return centers, radii, sum_r

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")

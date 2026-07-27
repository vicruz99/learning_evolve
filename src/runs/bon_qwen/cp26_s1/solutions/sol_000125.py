# sol_000125 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 69804dab) state=ec580d3d sum of radii=2.539289 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)

    # 1. Initialization: Best Next Point Heuristic
    centers = np.empty((n, 2))
    centers[0] = [0.5, 0.5]
    
    # Sample size for finding the best location
    num_samples = 10000
    
    for i in range(1, n):
        # Generate random candidate points
        candidates = np.random.rand(num_samples, 2)
        
        best_idx = 0
        max_min_dist = -1
        
        for idx in range(num_samples):
            c = candidates[idx]
            # Distance to boundaries
            d_bound = min(c[0], 1 - c[0], c[1], 1 - c[1])
            
            # Distance to existing centers
            if d_bound <= max_min_dist:
                continue
                
            # Optimization: compute distances
            dists = np.linalg.norm(centers[:i] - c, axis=1)
            d_circles = np.min(dists)
            
            min_dist = min(d_bound, d_circles)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        centers[i] = candidates[best_idx]

    # 2. Calculate initial uniform radius
    # Constraint 1: Distance to walls
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1])
    )
    r_wall = np.min(wall_dists)
    
    # Constraint 2: Distance between circles
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    r_pair = min_pair_dist / 2.0
    initial_r = min(r_wall, r_pair)
    
    # Initialize radii
    radii = np.full(n, initial_r)

    # 3. Optimization using SLSQP
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.hstack([centers.flatten(), radii])

    def objective(x):
        r = x[2::3]
        return -np.sum(r)

    def constraint_boundaries(x):
        c = x.reshape(-1, 3)
        # x - r >= 0
        con1 = c[:, 0] - c[:, 2]
        # 1 - x - r >= 0
        con2 = 1.0 - c[:, 0] - c[:, 2]
        # y - r >= 0
        con3 = c[:, 1] - c[:, 2]
        # 1 - y - r >= 0
        con4 = 1.0 - c[:, 1] - c[:, 2]
        return np.concatenate([con1, con2, con3, con4])

    def constraint_non_overlap(x):
        c = x.reshape(-1, 3)
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                # ||ci - cj||^2 - (ri + rj)^2 >= 0
                dist_sq = np.sum((c[i, :2] - c[j, :2])**2)
                rad_sum_sq = (c[i, 2] + c[j, 2])**2
                cons.append(dist_sq - rad_sum_sq)
        return np.array(cons)

    # Bounds: 0 <= x,y <= 1, 0 <= r <= 0.5
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundaries},
        {'type': 'ineq', 'fun': constraint_non_overlap}
    ]

    # Run optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    # Extract results
    x_opt = res.x.reshape(-1, 3)
    final_centers = x_opt[:, :2]
    final_radii = x_opt[:, 2]
    
    # Ensure non-negativity (fix potential numerical issues)
    final_radii = np.maximum(final_radii, 1e-9)
    
    # Ensure boundaries (fix potential numerical issues)
    final_centers[:, 0] = np.clip(final_centers[:, 0], final_radii, 1 - final_radii)
    final_centers[:, 1] = np.clip(final_centers[:, 1], final_radii, 1 - final_radii)

    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii

if __name__ == "__main__":
    import numpy as np
    
    # Re-define validate inside or assume it's available? 
    # The prompt says "We will run the below validation function", so I don't need to include it in my code,
    # but I can test it if I were running locally. 
    # For the final output, I just need run_packing.
    
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")

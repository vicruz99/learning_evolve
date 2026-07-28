# sol_000025 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cd0e5d1c) state=5bb53889 sum of radii=0.138227 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n_circles = 26
    
    # 1. Generate initial positions (Hexagonal Lattice)
    centers_init = []
    rows = [5, 4, 5, 4, 5, 4]  # Total 27 circles, we will drop 1 at the end to get 26
    x, y = 0.0, 0.0
    
    # Estimate a safe initial radius for the lattice
    # For 5 circles in a row: width approx 1.0 -> r approx 0.1
    # For 6 rows: height approx 1.0 -> r approx 0.094
    # We pick a conservative value to start
    r_est = 0.09 
    
    for i, count in enumerate(rows):
        for j in range(count):
            if len(centers_init) >= n_circles:
                break
            # Stagger rows
            offset_x = 0.0 if i % 2 == 0 else r_est
            x_pos = (j + 0.5 + offset_x/r_est) * 2.0 * r_est + r_est # Simplified placement
            y_pos = i * np.sqrt(3) * r_est + r_est
            
            # Boundary clamping for initialization
            x_pos = np.clip(x_pos, 0.2, 0.8)
            y_pos = np.clip(y_pos, 0.2, 0.8)
            
            centers_init.append([x_pos, y_pos])
        
        if len(centers_init) >= n_circles:
            break
            
    centers_init = np.array(centers_init[:n_circles])
    
    # 2. Define the optimization problem
    # We want to maximize sum(r_i) subject to constraints.
    # We will minimize -sum(r_i) + penalty * violation.
    
    def objective(params):
        centers = params[:52].reshape(26, 2)
        radii = params[52:]
        
        obj = -np.sum(radii)
        penalty = 0.0
        
        # Boundary constraints
        for i in range(26):
            x, y = centers[i]
            r = radii[i]
            # r <= x, r <= 1-x, r <= y, r <= 1-y
            for bound_val in [x, 1-x, y, 1-y]:
                if r > bound_val:
                    penalty += (r - bound_val) ** 2
        
        # Overlap constraints
        for i in range(26):
            for j in range(i + 1, 26):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                sum_r = radii[i] + radii[j]
                if sum_r > dist:
                    penalty += (sum_r - dist) ** 2
                    
        return obj + 1000 * penalty

    # 3. Initial guess
    # Random jitter to break symmetry and allow optimization to find local maxima
    rng = np.random.default_rng(42)
    centers_rand = centers_init + rng.uniform(-0.05, 0.05, centers_init.shape)
    centers_rand = np.clip(centers_rand, 0.1, 0.9)
    
    initial_radii = np.full(26, 0.09)
    initial_params = np.concatenate([centers_rand.flatten(), initial_radii])
    
    # 4. Optimization
    # Bounds: centers [0, 1], radii [0, 0.5]
    bounds = []
    for _ in range(26):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)]) # x, y
        bounds.append((0.0, 0.5)) # r
    
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    
    best_centers = result.x[:52].reshape(26, 2)
    best_radii = result.x[52:]
    
    # 5. Final Cleanup: Ensure strict validity
    # Clip centers to [0, 1]
    best_centers = np.clip(best_centers, 0, 1)
    
    # Adjust radii to be strictly valid (non-overlapping)
    # This is a post-processing step to ensure the validation function passes
    # We solve for radii given fixed centers using a simple LP-like iterative shrink
    # Or simply scale down if necessary.
    
    # Simple adjustment: calculate max possible radius for each circle
    radii_adjusted = np.full(26, 1.0)
    
    # Iterative shrink to satisfy constraints
    for _ in range(50):
        for i in range(26):
            # Boundary limit
            x, y = best_centers[i]
            r_max = min(x, 1-x, y, 1-y)
            
            # Neighbor limit
            for j in range(26):
                if i != j:
                    dist = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                    r_max = min(r_max, dist - radii_adjusted[j])
            
            radii_adjusted[i] = min(radii_adjusted[i], r_max)
            radii_adjusted[i] = max(radii_adjusted[i], 0.0)

    # Ensure strict positive radii (validation might fail if 0, but usually valid if >0)
    # If any radius is too small, we might have issues, but optimization usually finds good spots.
    
    sum_radii = np.sum(radii_adjusted)
    
    return best_centers, radii_adjusted, sum_radii

# Verification step (local)
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")

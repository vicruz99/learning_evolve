# sol_000101 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 15bab5cf) state=a1ec2c8b sum of radii=0.179954 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hybrid Hexagonal Pattern
    # Rows configuration: 6, 5, 6, 5, 4 circles
    rows_config = [6, 5, 6, 5, 4]
    centers = []
    r_est = 0.09 # Initial estimate
    
    y_offset = 0.0
    row_height = r_est * np.sqrt(3)
    
    for i, count in enumerate(rows_config):
        y = r_est + i * row_height
        # Stagger odd rows (1st, 3rd, etc. in 0-indexed config)
        # Actually, let's stagger based on parity to fit better
        if i % 2 == 1:
            x_start = r_est + r_est
        else:
            x_start = r_est
            
        # If row has 6 circles, it needs width 12r. If 5 circles, width 10r.
        # We will adjust centers dynamically in optimization, but start evenly
        width_avail = 1.0 - 2 * r_est
        if count > 0:
            spacing = width_avail / (count - 1) if count > 1 else 0
        else:
            spacing = 0
            
        for j in range(count):
            x = x_start + j * spacing
            centers.append([x, y])
            
    centers = np.array(centers)
    radii = np.ones(n) * r_est

    # 2. Optimization: Repulsive Force / Gradient Ascent Hybrid
    # We maximize sum of radii by penalizing overlaps and boundary violations.
    # Using a penalty method allows us to handle the non-convex constraints robustly.
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...] -> flatten centers and radii
    # Actually, keeping them separate for the solver is cleaner.
    # We will use a custom objective with L-BFGS-B or similar.
    
    def objective(vars_1d):
        c = vars_1d[:2 * n].reshape(n, 2)
        r = vars_1d[2 * n:]
        
        # Objective: Maximize sum of radii (minimize negative sum)
        obj = -np.sum(r)
        
        # Penalty parameters
        penalty = 1000.0
        eps = 1e-4
        
        # Boundary constraints
        # x - r >= 0  => x < r is violation
        # x + r <= 1  => x + r > 1 is violation
        penalty += penalty * np.sum(np.maximum(0, r - c[:, 0])**2)
        penalty += penalty * np.sum(np.maximum(0, c[:, 0] + r - 1)**2)
        penalty += penalty * np.sum(np.maximum(0, r - c[:, 1])**2)
        penalty += penalty * np.sum(np.maximum(0, c[:, 1] + r - 1)**2)
        
        # Overlap constraints
        # ||ci - cj|| >= ri + rj
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j])**2) + eps)
                overlap = (r[i] + r[j] - dist)
                if overlap > 0:
                    penalty += penalty * overlap**2
                    
        return penalty

    # Initial guess
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: centers in [0, 1], radii in [0, 0.5]
    bnds = []
    for i in range(n):
        bnds.append((0.0, 1.0)) # x
        bnds.append((0.0, 1.0)) # y
        bnds.append((0.0, 0.5)) # r

    # Run optimization
    # Using Powell or L-BFGS-B. L-BFGS-B is good with bounds.
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bnds, options={'maxiter': 2000, 'ftol': 1e-9})
    
    best_centers = res.x[:2 * n].reshape(n, 2)
    best_radii = res.x[2 * n:]
    
    # 3. Refinement: Ensure strict validity and clamp radii
    # Sometimes the optimizer pushes radii too high slightly due to penalty tolerance.
    # We will compute the exact max radius for each circle given the final centers.
    
    final_radii = np.zeros(n)
    for i in range(n):
        c = best_centers[i]
        r_boundary = min(c[0], 1 - c[0], c[1], 1 - c[1])
        r_overlap = 1.0 # Large initial
        
        for j in range(n):
            if i == j: continue
            dist = np.sqrt(np.sum((c - best_centers[j])**2))
            # We don't know final_radii[j] yet. 
            # A safe heuristic for refinement is to take the current optimized radius
            # but ensure it doesn't violate neighbors' current radii.
            # However, for a valid output, we can just take the optimized radii
            # and scale them down slightly if needed, or just trust the optimizer.
            pass
        final_radii[i] = best_radii[i]

    # Final check and slight safety scaling if needed
    # The penalty method usually finds a valid config if penalty is high enough.
    # Let's verify and adjust if any overlap exists.
    # If the optimizer result is valid, we return it.
    
    # To be absolutely safe against numerical drift, we can compute the "tightest" valid radii
    # based on the optimized centers.
    radii_final = np.zeros(n)
    for i in range(n):
        # Max radius allowed by boundaries
        r_lim = min(best_centers[i, 0], 1 - best_centers[i, 0], 
                    best_centers[i, 1], 1 - best_centers[i, 1])
        
        # Max radius allowed by neighbors
        # Since radii are coupled, we use the optimized radii as a baseline
        # and ensure they satisfy constraints.
        r_val = best_radii[i]
        
        # Check against all neighbors
        for j in range(n):
            if i == j: continue
            dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
            # Constraint: r_i + r_j <= dist
            # r_i <= dist - r_j
            if dist - best_radii[j] < r_val:
                r_val = dist - best_radii[j]
        
        r_val = min(r_val, r_lim)
        radii_final[i] = max(0, r_val)

    # Recalculate sum
    sum_r = np.sum(radii_final)
    
    # Final sanity check for overlaps (should be none)
    # If radii were reduced to be valid, sum might be lower, but config is valid.
    
    return best_centers, radii_final, sum_r

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")

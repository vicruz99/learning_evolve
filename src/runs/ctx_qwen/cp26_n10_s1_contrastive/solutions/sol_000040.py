# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=b2bf3834 sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[2 * N_CIRCLES:])

def constraints(vars):
    """
    Constraint function: ensures circles stay within bounds and do not overlap.
    Returns a 1D array of constraint values that must be >= 0.
    """
    centers = vars[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = vars[2 * N_CIRCLES:]

    c_list = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c_list.append(centers[:, 0] - radii)
    c_list.append(1.0 - centers[:, 0] - radii)
    c_list.append(centers[:, 1] - radii)
    c_list.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    # Vectorized computation for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Only need upper triangular part (i < j) to avoid duplicates and self-comparison
    i, j = np.triu_indices(N_CIRCLES, k=1)
    c_list.append(dist_sq[i, j] - r_sum[i, j]**2)
    
    return np.concatenate(c_list)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -1.0
    best_vars = None
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Known optimal topology for 26 circles: hexagonal rows with counts 6, 5, 6, 5, 4
    row_counts = [6, 5, 6, 5, 4]
    
    # Multi-start optimization
    for seed in range(30):
        np.random.seed(seed)
        
        # Estimate a safe initial radius for the hex layout
        r_init = 0.092
        
        centers = []
        y = r_init
        for row_idx, count in enumerate(row_counts):
            # Shift odd rows horizontally by one radius to form hexagonal packing
            x_start = 2 * r_init if row_idx % 2 == 1 else r_init
            x = x_start
            for _ in range(count):
                centers.append([x, y])
                x += 2 * r_init
            y += np.sqrt(3.0) * r_init
            
        centers = np.array(centers)
        
        # Add random perturbation to break symmetry and explore landscape
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        vars0 = np.concatenate([centers.flatten(), np.full(N_CIRCLES, r_init)])
        
        try:
            res = minimize(
                objective, 
                vars0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 4000, 'ftol': 1e-12}
            )
            
            if res.success:
                curr_sum = np.sum(res.x[2 * N_CIRCLES:])
                # Check if constraints are satisfied within numerical tolerance
                min_c = np.min(constraints(res.x))
                if min_c >= -1e-7 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue

    # Fallback to a valid grid packing if optimization fails entirely
    if best_vars is None:
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
        pts.append([0.5, 0.5])
        centers = np.array(pts)
        radii = np.full(N_CIRCLES, 0.09)
        best_vars = np.concatenate([centers.flatten(), radii])
        best_sum = np.sum(radii)

    # Final high-precision refinement on the best configuration found
    try:
        res = minimize(
            objective, 
            best_vars, 
            method='SLSQP', 
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 8000, 'ftol': 1e-14}
        )
        if np.min(constraints(res.x)) >= -1e-7:
            best_vars = res.x
            best_sum = np.sum(best_vars[2 * N_CIRCLES:])
    except Exception:
        pass

    # Extract and format results
    centers = best_vars[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_vars[2 * N_CIRCLES:]
    
    # Ensure radii are strictly non-negative
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(best_sum)

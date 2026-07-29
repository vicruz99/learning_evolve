# sol_000302 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d28721c0) state=78795b81 sum of radii=2.610068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers in a structured, hexagonal-inspired pattern
    # Row distribution: 6, 5, 6, 5, 4 circles (total 26)
    row_counts = [6, 5, 6, 5, 4]
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.06)  # Starting radius guess
    idx = 0
    for i, count in enumerate(row_counts):
        y = (i + 0.5) / 5.0
        for j in range(count):
            x = (j + 0.5) / (count + 1)
            centers[idx] = [x, y]
            idx += 1
            
    # Add slight random perturbation to break symmetry and aid optimization
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # Flatten variables for scipy: [x1, y1, x2, y2, ..., x26, y26, r1, r2, ..., r26]
    x0 = np.concatenate([centers.flatten(), radii])
    
    def objective(vars):
        # Maximize sum of radii => minimize negative sum
        return -np.sum(vars[2*n:])
        
    def con_func(vars):
        c = vars[:2*n].reshape((n, 2))
        r = vars[2*n:]
        n_con = 4*n + n*(n-1)//2
        vals = np.empty(n_con)
        k = 0
        
        # Boundary constraints: c - r >= 0  and  1 - (c + r) >= 0
        for i in range(n):
            vals[k] = c[i, 0] - r[i]; k+=1
            vals[k] = 1.0 - (c[i, 0] + r[i]); k+=1
            vals[k] = c[i, 1] - r[i]; k+=1
            vals[k] = 1.0 - (c[i, 1] + r[i]); k+=1
            
        # Pairwise non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
        # Vectorized computation for speed
        dx = c[:, 0, np.newaxis] - c[np.newaxis, :, 0]
        dy = c[:, 1, np.newaxis] - c[np.newaxis, :, 1]
        dist_sq = dx*dx + dy*dy
        
        tri_idx = np.tril_indices(n, -1)  # Lower triangle, excluding diagonal
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        vals[k:] = dist_sq[tri_idx] - r_sum[tri_idx]**2
        return vals

    con = NonlinearConstraint(con_func, 0, np.inf)
    bounds = [(0.0, 1.0)]*(2*n) + [(0.0, None)]*n
    
    # Run SLSQP optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=con,
                   options={'maxiter': 2000, 'ftol': 1e-12})
                   
    final_centers = res.x[:2*n].reshape((n, 2))
    final_radii = res.x[2*n:]
    
    # Safety clamping to guarantee strict boundary compliance within tolerance
    final_centers[:, 0] = np.clip(final_centers[:, 0], final_radii, 1.0 - final_radii)
    final_centers[:, 1] = np.clip(final_centers[:, 1], final_radii, 1.0 - final_radii)
    
    return final_centers, final_radii, float(np.sum(final_radii))

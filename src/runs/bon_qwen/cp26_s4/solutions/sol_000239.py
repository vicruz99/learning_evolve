# sol_000239 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 06f8ea92) state=207d988a sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def _compute_constraints(v):
    """
    Compute all inequality constraints for the packing problem.
    Returns a 1D numpy array of constraint values (must be >= 0).
    """
    centers = v[:2 * N_CIRCLES].reshape((N_CIRCLES, 2))
    radii = v[2 * N_CIRCLES:]
    
    cons = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    for i in range(N_CIRCLES):
        cons.append(centers[i, 0] - radii[i])
        cons.append(1.0 - centers[i, 0] - radii[i])
        cons.append(centers[i, 1] - radii[i])
        cons.append(1.0 - centers[i, 1] - radii[i])
        
    # Overlap constraints using vectorized broadcasting for efficiency
    # Shape: (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract upper triangle indices (i < j)
    upper_tri_indices = np.triu_indices(N_CIRCLES, k=1)
    overlap_vals = dist_sq[upper_tri_indices] - rad_sum[upper_tri_indices]**2
    cons.extend(overlap_vals)
    
    return np.array(cons)

def _run_optimization(initial_centers, initial_radii):
    """Run SLSQP optimization from a given initial state."""
    n = N_CIRCLES
    x0 = np.hstack([initial_centers.ravel(), initial_radii])
    
    # Variable bounds: centers in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    def objective(v):
        return -np.sum(v[2 * n:])
        
    cons = {'type': 'ineq', 'fun': _compute_constraints}
    
    res = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False}
    )
    return res

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate a hexagonal initial configuration
    # Pattern: 5, 4, 5, 4, 5, 3 circles per row
    row_counts = [5, 4, 5, 4, 5, 3]
    r_base = 0.09  # Base spacing radius
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.04)  # Start with safe small radius
    
    idx = 0
    for k, count in enumerate(row_counts):
        y = r_base + k * r_base * np.sqrt(3)
        for m in range(count):
            if idx >= n:
                break
            x = r_base + m * 2 * r_base + (k % 2) * r_base
            centers[idx] = [x, y]
            idx += 1
        if idx >= n:
            break
            
    # Run multiple trials with perturbations to find global optimum
    np.random.seed(42)
    for trial in range(5):
        # Perturb initial state
        c_trial = centers + np.random.rand(n, 2) * 0.015
        r_trial = radii + np.random.rand(n) * 0.008
        
        # Clamp to feasible region for optimizer start
        c_trial = np.clip(c_trial, 0.05, 0.95)
        r_trial = np.clip(r_trial, 0.02, 0.12)
        
        res = _run_optimization(c_trial, r_trial)
        
        if res.success:
            curr_sum = -res.fun
            if curr_sum > best_sum:
                # Verify constraints manually to be safe against numerical slips
                final_c = res.x[:2 * n].reshape((n, 2))
                final_r = res.x[2 * n:]
                
                # Quick validity check
                valid = True
                if np.any(final_r < 0) or np.any(np.isnan(final_c)) or np.any(np.isnan(final_r)):
                    valid = False
                else:
                    # Check bounds & overlaps
                    for i in range(n):
                        if final_c[i,0] - final_r[i] < -1e-9 or final_c[i,0] + final_r[i] > 1 + 1e-9:
                            valid = False; break
                        if final_c[i,1] - final_r[i] < -1e-9 or final_c[i,1] + final_r[i] > 1 + 1e-9:
                            valid = False; break
                    if valid:
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = np.linalg.norm(final_c[i] - final_c[j])
                                if d < final_r[i] + final_r[j] - 1e-9:
                                    valid = False; break
                            if not valid: break
                                
                if valid:
                    best_sum = curr_sum
                    best_centers = final_c
                    best_radii = final_r

    # Fallback to initial valid config if optimization fails entirely
    if best_centers is None:
        best_centers, best_radii = centers, radii
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum

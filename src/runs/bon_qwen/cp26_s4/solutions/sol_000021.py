# sol_000021 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d7881bb) state=54211e78 sum of radii=2.542779 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(vars):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars[2::3])

def get_constraints(N):
    """Generate boundary and non-overlap constraints for SLSQP."""
    cons = []
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2 - (x[3*i+2]+x[3*j+2])**2})
    return cons

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    np.random.seed(42)
    centers = np.random.rand(N, 2) * 0.8 + 0.1
    radii = np.full(N, 0.02)
    
    alpha = 0.1
    # Stage 1: Force-directed relaxation to expand circles
    for step in range(2000):
        # Vectorized pairwise forces
        diffs = centers[:, None, :] - centers[None, :, :]          # (N, N, 2)
        dists = np.linalg.norm(diffs, axis=2)                      # (N, N)
        np.fill_diagonal(dists, np.inf)
        target_dists = radii[:, None] + radii[None, :]             # (N, N)
        overlap = np.maximum(0, target_dists - dists)              # (N, N)
        dirs = diffs / np.maximum(dists[:, :, None], 1e-6)         # (N, N, 2)
        forces = np.sum(overlap[:, :, None] * dirs, axis=1)        # (N, 2)
        
        # Boundary repulsion forces
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x < r: forces[i, 0] += 0.5 * (r - x) / max(r, 0.01)
            if x > 1 - r: forces[i, 0] -= 0.5 * (x - (1 - r)) / max(r, 0.01)
            if y < r: forces[i, 1] += 0.5 * (r - y) / max(r, 0.01)
            if y > 1 - r: forces[i, 1] -= 0.5 * (y - (1 - r)) / max(r, 0.01)
            
        centers += alpha * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        max_ov = np.max(overlap)
        if max_ov < 1e-4:
            radii *= 1.002  # Grow if stable
        elif max_ov > 0.1:
            radii *= 0.99   # Shrink if heavy overlap
            
        alpha *= 0.998  # Anneal step size
        
    # Stage 2: SLSQP refinement
    x0 = np.zeros(3*N)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    constraints = get_constraints(N)
    bnds = [(None, None)] * (3*N)
    for i in range(N): bnds[3*i+2] = (1e-6, None)
        
    res = minimize(obj_func, x0, method='SLSQP', bounds=bnds, 
                   constraints=constraints, options={'maxiter': 300, 'ftol': 1e-9})
    
    final_centers = res.x.reshape(N, 3)[:, :2]
    final_radii = res.x.reshape(N, 3)[:, 2]
    
    # Ensure strict feasibility
    final_radii = np.clip(final_radii, 0, None)
    final_centers = np.clip(final_centers, final_radii[:, None], 1 - final_radii[:, None])
    
    return final_centers, final_radii, float(np.sum(final_radii))

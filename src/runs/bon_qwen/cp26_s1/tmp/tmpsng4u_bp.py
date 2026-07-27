import numpy as np
from scipy.optimize import differential_evolution

N = 26

def compute_objective(params):
    """
    Objective function: maximize sum of radii with penalties for invalid configurations.
    params: flattened array of [x1, y1, ..., x26, y26, r1, ..., r26]
    """
    centers = params[:2*N].reshape(N, 2)
    radii = params[2*N:]
    
    # Boundary penalties: circles must be within [0, 1]^2
    # Violation occurs if r > x, r > 1-x, r > y, or r > 1-y
    v1 = np.maximum(0, radii - centers[:, 0])
    v2 = np.maximum(0, radii - (1 - centers[:, 0]))
    v3 = np.maximum(0, radii - centers[:, 1])
    v4 = np.maximum(0, radii - (1 - centers[:, 1]))
    pen_b = np.sum(v1**2 + v2**2 + v3**2 + v4**2)
    
    # Overlap penalties: distance between centers must be >= sum of radii
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Only consider upper triangle to avoid double counting
    i, j = np.triu_indices(N, k=1)
    overlaps = np.maximum(0, r_sum[i, j] - dist[i, j])
    pen_o = np.sum(overlaps**2)
    
    # Large weight ensures validity is prioritized over radius sum
    return -np.sum(radii) + 1e4 * (pen_b + pen_o)

def run_packing():
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.001, 0.5)] * N
    
    # Global optimization using Differential Evolution
    result = differential_evolution(
        compute_objective, 
        bounds, 
        seed=42, 
        popsize=50, 
        maxiter=2000, 
        tol=1e-9, 
        polish=True
    )
    
    centers = result.x[:2*N].reshape(N, 2)
    radii = result.x[2*N:]
    
    # Post-processing to guarantee strict validity within numerical tolerance
    for _ in range(100):
        # Boundary slacks
        slacks_b = np.minimum(
            np.minimum(centers[:, 0] - radii, 1 - centers[:, 0] - radii),
            np.minimum(centers[:, 1] - radii, 1 - centers[:, 1] - radii)
        )
        min_slack = np.min(slacks_b)
        
        # Overlap slacks
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        slacks_o = dists - r_sums
        i, j = np.triu_indices(N, k=1)
        min_slack = min(min_slack, np.min(slacks_o[i, j]))
        
        # If any constraint is violated, shrink radii uniformly to fix it
        if min_slack < -1e-8:
            shrink = -min_slack + 1e-9
            radii = np.maximum(radii - shrink, 0.0)
        else:
            break
            
    return centers, radii, np.sum(radii)
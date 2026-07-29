# sol_000284 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5952a474) state=2dcbd9e8 sum of radii=1.509968 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_bottleneck(centers):
    """
    Computes the maximum feasible radius for a given set of centers.
    This is the minimum distance to any boundary or to any other circle (divided by 2).
    """
    n = centers.shape[0]
    # Distance to boundaries
    d_bound = np.min(np.concatenate([
        centers[:, 0], 1 - centers[:, 0],
        centers[:, 1], 1 - centers[:, 1]
    ]))
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    d_pair = np.min(dists) / 2.0
    
    return min(d_bound, d_pair)

def objective(coords, n):
    """Objective function to minimize (negative bottleneck distance)."""
    centers = coords.reshape(n, 2)
    return -compute_bottleneck(centers)

def run_packing():
    n = 26
    np.random.seed(42)
    
    # 1. Initialization: Random placement inside the square
    centers = np.random.uniform(0.1, 0.9, size=(n, 2))
    
    # 2. Force-Directed Relaxation
    # Resolves overlaps and pushes circles into a stable configuration quickly.
    r_init = 0.05
    for _ in range(3000):
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r_init and dist > 1e-8:
                    push = (2 * r_init - dist) * diff / dist
                    forces[i] += push
                    forces[j] -= push
            # Boundary repulsion
            for d in range(2):
                if centers[i, d] < r_init:
                    forces[i, d] += (r_init - centers[i, d]) * 10
                elif centers[i, d] > 1 - r_init:
                    forces[i, d] -= (centers[i, d] - (1 - r_init)) * 10
        centers += forces * 0.5
        centers = np.clip(centers, 0, 1)
        
    x0 = centers.flatten()
    
    # 3. Local Optimization
    # Maximizes the bottleneck distance using derivative-free Nelder-Mead.
    res = minimize(
        objective, 
        x0, 
        args=(n,), 
        method='Nelder-Mead', 
        options={'maxiter': 50000, 'xatol': 1e-7, 'fatol': 1e-9}
    )
    
    best_centers = res.x.reshape(n, 2)
    r_opt = compute_bottleneck(best_centers)
    
    # 4. Safety margin to satisfy validator's 1e-12 tolerance
    r_final = r_opt * 0.99999
    radii = np.full(n, r_final)
    
    return best_centers, radii, 26 * r_final

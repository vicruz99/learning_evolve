# sol_000053 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bd759b5e) state=015a3cfa sum of radii=2.466622 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x):
    n = 26
    centers = x.reshape(n, 2)
    
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    bounds_dist = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dists = np.min(dists, axis=1)
    
    # Feasible radius for each circle
    radii = np.minimum(bounds_dist, min_pair_dists / 2.0)
    return -np.sum(radii)

def get_radii(centers):
    n = centers.shape[0]
    bounds_dist = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dists = np.min(dists, axis=1)
    return np.minimum(bounds_dist, min_pair_dists / 2.0)

def run_packing():
    n = 26
    best_val = np.inf
    best_centers = None
    
    # Initial guess 1: 5x5 grid + center circle
    c1 = []
    for i in range(5):
        for j in range(5):
            c1.append([0.1 + 0.2*i, 0.1 + 0.2*j])
    c1.append([0.5, 0.5])
    c1 = np.array(c1)
    
    # Initial guess 2: Hexagonal pattern scaled to fit
    c2 = []
    scale = 0.115
    for row in range(6):
        y = 0.05 + row * scale * np.sqrt(3)
        if y > 0.95:
            break
        shift = (scale / 2.0) if row % 2 == 1 else 0.0
        n_in_row = 6 if row % 2 == 0 else 5
        for col in range(n_in_row):
            if len(c2) < 26:
                x = 0.05 + shift + col * scale * 2.0
                if x <= 0.95:
                    c2.append([x, y])
    c2 = np.array(c2[:26])
    
    # Collect initial guesses
    guesses = [c1, c2]
    
    # Add randomized perturbations of structured guesses
    np.random.seed(42)
    for _ in range(12):
        for base in [c1, c2]:
            pert = base.copy()
            pert += np.random.uniform(-0.03, 0.03, pert.shape)
            pert = np.clip(pert, 0.02, 0.98)
            guesses.append(pert)
            
    # Optimize from each guess
    for init_centers in guesses:
        x0 = init_centers.flatten()
        try:
            res = minimize(objective, x0, 
                          method='Nelder-Mead', 
                          options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(n, 2)
        except Exception:
            continue
            
    if best_centers is None:
        best_centers = c1
        
    # Compute final valid radii from optimized centers
    final_radii = get_radii(best_centers)
    total_sum = np.sum(final_radii)
    
    return best_centers, final_radii, total_sum

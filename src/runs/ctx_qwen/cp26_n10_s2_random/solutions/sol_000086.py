# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000063 (state 0dfa75ae) state=99c2e151 sum of radii=1.862508 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, differential_evolution

def compute_radii(centers):
    """Computes the maximum valid radius for each circle given fixed centers."""
    n = centers.shape[0]
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                         np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Radius limited by half the distance to the nearest neighbor
    r_pair = 0.5 * np.min(dists, axis=1)
    return np.minimum(r_bound, r_pair)

def objective(x):
    """Objective function for minimization: maximize sum of radii."""
    centers = x.reshape(-1, 2)
    radii = compute_radii(centers)
    return -np.sum(radii)

def get_lattice_starts(n):
    """Generates diverse hexagonal and square lattice initial configurations."""
    starts = []
    # Hexagonal lattices with various densities
    for r0 in np.linspace(0.08, 0.11, 6):
        pts = []
        y = r0
        row = 0
        while len(pts) < n:
            x = r0 if row % 2 == 0 else 2 * r0
            while x + r0 <= 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        starts.append(np.array(pts[:n]).flatten())
    # Square grids
    for r0 in np.linspace(0.08, 0.105, 5):
        pts = []
        y = r0
        row = 0
        while len(pts) < n:
            x = r0
            while x + r0 <= 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2 * r0
            y += 2 * r0
            row += 1
        starts.append(np.array(pts[:n]).flatten())
    return starts

def run_packing():
    np.random.seed(42)
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n)
    
    best_val = np.inf
    best_centers = None
    
    # Phase 1: Lattice-based local optimizations with Powell
    for x0 in get_lattice_starts(n):
        x0 = np.clip(x0 + np.random.normal(0, 1e-4, size=x0.shape), 0.0, 1.0)
        try:
            res = minimize(objective, x0, method='Powell', bounds=bounds,
                           options={'maxiter': 1500, 'ftol': 1e-13, 'xtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(-1, 2)
        except Exception:
            pass
            
    # Phase 2: Global search with Differential Evolution
    # Explores non-lattice configurations and complex basins effectively
    try:
        de_res = differential_evolution(objective, bounds, popsize=20, maxiter=60,
                                        mutation=(0.6, 1.2), recombination=0.9,
                                        seed=42, tol=1e-8, polish=True)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_centers = de_res.x.reshape(-1, 2)
    except Exception:
        pass
        
    # Phase 3: Iterative perturbation & refinement to escape shallow local minima
    if best_centers is not None:
        curr = best_centers.copy()
        for step in range(25):
            pert = curr + np.random.normal(0, 0.0015, curr.shape)
            pert = np.clip(pert, 0.001, 0.999)
            try:
                res = minimize(objective, pert.flatten(), method='Powell', bounds=bounds,
                               options={'maxiter': 1000, 'ftol': 1e-13})
                if res.fun < best_val:
                    best_val = res.fun
                    curr = res.x.reshape(-1, 2)
            except Exception:
                pass
        best_centers = curr
        
    final_radii = compute_radii(best_centers)
    
    # Final safety clamp to guarantee validation passes within tolerance
    for i in range(n):
        x, y = best_centers[i]
        r = final_radii[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r + 1e-9:
            final_radii[i] = max_r
            
    final_radii = np.maximum(final_radii, 0.0)
    
    return best_centers, final_radii, float(np.sum(final_radii))

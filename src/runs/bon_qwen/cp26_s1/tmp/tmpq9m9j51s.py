import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 26

def objective_func(X):
    """Objective: maximize sum of radii (minimize negative sum)."""
    return -np.sum(X[2 * N:])

def constraint_func(X):
    """Returns all inequality constraints >= 0."""
    centers = X[:2 * N].reshape(N, 2)
    radii = X[2 * N:]

    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = centers[:, 0] - radii
    c2 = 1.0 - centers[:, 0] - radii
    c3 = centers[:, 1] - radii
    c4 = 1.0 - centers[:, 1] - radii

    # Pairwise non-overlap constraints: dist_ij >= r_i + r_j
    dists = squareform(pdist(centers))
    r_sum = radii[:, None] + radii[None, :]
    idx = np.triu_indices(N, k=1)
    c_pairwise = dists[idx] - r_sum[idx]

    return np.concatenate([c1, c2, c3, c4, c_pairwise])

def run_packing():
    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    constraints = {'type': 'ineq', 'fun': constraint_func}

    best_X = None
    best_obj = np.inf  # We minimize negative sum, so track minimum

    # Try multiple seeds to escape local optima
    for seed in range(10):
        rng = np.random.RandomState(seed)
        
        # Initialize centers on a perturbed grid
        centers_init = np.zeros((N, 2))
        radii_init = np.ones(N) * 0.08
        
        k = 0
        for r in range(6):
            for c in range(5):
                if k < N:
                    centers_init[k, 0] = 0.2 * (c + 0.5) + rng.uniform(-0.04, 0.04)
                    centers_init[k, 1] = 0.2 * (r + 0.5) + rng.uniform(-0.04, 0.04)
                    k += 1
        
        # Fill remaining if any
        while k < N:
            centers_init[k, 0] = rng.uniform(0.1, 0.9)
            centers_init[k, 1] = rng.uniform(0.1, 0.9)
            k += 1

        X0 = np.concatenate([centers_init.flatten(), radii_init])
        # Ensure X0 respects bounds strictly
        for i in range(2 * N):
            X0[i] = np.clip(X0[i], 0.0, 1.0)

        res = minimize(objective_func, X0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-10})

        # Check feasibility and improve best
        if np.min(constraint_func(res.x)) >= -1e-7:
            if res.fun < best_obj:
                best_obj = res.fun
                best_X = res.x.copy()

    # Fallback if optimization failed unexpectedly
    if best_X is None:
        best_X = np.concatenate([centers_init.flatten(), radii_init])

    centers = best_X[:2 * N].reshape(N, 2)
    radii = best_X[2 * N:]
    
    # Post-processing: ensure strict feasibility within validation tolerance
    min_c = np.min(constraint_func(best_X))
    if min_c < -1e-9:
        # Shrink radii proportionally to fix violations
        safety = 1.0 - np.abs(min_c) * 15.0
        radii *= np.max([safety, 0.99])
        
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
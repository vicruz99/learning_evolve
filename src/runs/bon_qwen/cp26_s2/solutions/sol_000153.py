# sol_000153 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 46a34d55) state=c259bea5 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    """Maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2*N:])

def con_bounds(x):
    """Circles must stay inside [0,1]x[0,1]"""
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    return np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])

def con_pairs(x):
    """Circles must not overlap"""
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(d**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.triu(dist - r[:, np.newaxis] - r[np.newaxis, :], k=1).ravel()

def get_initial_guess(seed):
    """Generate a hexagonal-like initial placement with slight perturbation"""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 2))
    rows = [6, 5, 6, 5, 4]  # Sums to 26
    idx = 0
    for i, cnt in enumerate(rows):
        y = 0.12 + i * 0.19
        for j in range(cnt):
            x = 0.12 + j * 0.19 + (i % 2) * 0.10
            if idx < N:
                centers[idx] = [x, y]
                idx += 1
    # Add small noise to break symmetry and help optimizer
    centers += rng.normal(0, 0.005, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    x0 = np.zeros(3*N)
    x0[:2*N] = centers.flatten()
    x0[2*N:] = 0.09  # Initial radii guess
    return x0

def run_packing():
    best_res = None
    best_val = -np.inf
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N

    # Multiple restarts to ensure robust convergence
    for seed in range(3):
        x0 = get_initial_guess(seed)
        cons = [
            {'type': 'ineq', 'fun': con_bounds},
            {'type': 'ineq', 'fun': con_pairs}
        ]
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            continue

    if best_res is not None:
        centers = best_res.x[:2*N].reshape(N, 2)
        radii = best_res.x[2*N:]
        # Strict enforcement of bounds to pass validator comfortably
        radii = np.maximum(radii, 0.0)
        centers = np.clip(centers, radii[:, None], 1.0 - radii[:, None])
        return centers, radii, np.sum(radii)
    else:
        # Fallback (should not be reached)
        return np.zeros((N, 2)), np.zeros(N), 0.0

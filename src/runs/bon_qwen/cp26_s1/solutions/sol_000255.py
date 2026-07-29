# sol_000255 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3be09fa9) state=7c0ba890 sum of radii=2.477764 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constants to avoid closures
N_CIRCLES = 26
# Precompute pairwise indices for non-overlap constraints (i < j)
_PAIR_INDICES = np.argwhere(np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1))

def _objective(v):
    """Maximize the common radius r."""
    return -v[-1]

def _constraints(v):
    """
    Returns array of constraint values >= 0.
    Constraints: boundary margins and pairwise distances.
    """
    r = v[-1]
    centers = v[:-1].reshape(N_CIRCLES, 2)
    x, y = centers[:, 0], centers[:, 1]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_boundary = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise constraints: dist^2 >= (2r)^2
    i_idx, j_idx = _PAIR_INDICES[:, 0], _PAIR_INDICES[:, 1]
    diffs = centers[i_idx] - centers[j_idx]
    dists_sq = np.sum(diffs**2, axis=1)
    c_pairwise = dists_sq - (2.0 * r)**2
    
    return np.concatenate([c_boundary, c_pairwise])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.RandomState(42)
    best_r = 0.0
    best_centers = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Dense grid (6x5 subset)
    gx = np.linspace(0.1, 0.9, 6)
    gy = np.linspace(0.1, 0.9, 5)
    grid = np.array([(x, y) for x in gx for y in gy][:N_CIRCLES])
    inits.append(grid)
    
    # 2. Hexagonal-like arrangement
    hexc = []
    for i in range(6):
        for j in range(5):
            x = 0.1 + j * 0.18 + (i % 2) * 0.09
            y = 0.1 + i * 0.16
            if x <= 0.95 and y <= 0.95:
                hexc.append((x, y))
    if len(hexc) >= N_CIRCLES:
        inits.append(np.array(hexc[:N_CIRCLES]))
        
    # 3. Random perturbations of grid
    for _ in range(4):
        c = grid + rng.uniform(-0.02, 0.02, size=grid.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)

    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.2)]
    r_start = 0.045
    
    for init_c in inits:
        x0 = np.concatenate([init_c.ravel(), [r_start]])
        
        try:
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': _constraints},
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
            if res.success and res.x[-1] > best_r:
                best_r = res.x[-1]
                best_centers = res.x[:-1].reshape(N_CIRCLES, 2)
        except Exception:
            continue
            
    # Fallback if optimizer fails (highly unlikely with these inits)
    if best_centers is None:
        best_centers = inits[0]
        best_r = 0.05
        
    radii = np.full(N_CIRCLES, best_r)
    sum_radii = best_r * N_CIRCLES
    return best_centers, radii, sum_radii

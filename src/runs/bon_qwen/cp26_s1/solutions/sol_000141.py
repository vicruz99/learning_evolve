# sol_000141 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c86e033) state=d04eabad sum of radii=1.729642 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def _compute_min_clearance(vars, n):
    """Compute the minimum distance to other circles and boundaries."""
    pts = vars.reshape(n, 2)
    # Distances to boundaries
    clear = np.minimum(np.minimum(pts[:, 0], 1 - pts[:, 0]), 
                       np.minimum(pts[:, 1], 1 - pts[:, 1]))
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Minimum clearance overall
    return np.min(np.minimum(clear, np.min(dists, axis=1)))

def _objective(vars):
    """Objective function to minimize (negative of min clearance)."""
    return -_compute_min_clearance(vars, 26)

def _solve_radii(positions, n):
    """Solve LP to maximize sum of radii given fixed positions."""
    c = -np.ones(n)  # Maximize sum(r_i) => Minimize -sum(r_i)
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
        # Boundary constraints: r_i <= clearance
        bounds_i = [positions[i, 0], 1 - positions[i, 0], 
                    positions[i, 1], 1 - positions[i, 1]]
        for b in bounds_i:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds_r = [(0, None) for _ in range(n)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    return res.x if res.success else np.full(n, 0.0)

def run_packing():
    n = 26
    positions = np.zeros((n, 2))
    idx = 0
    # Initialize with a grid pattern
    for r in range(5):
        for c in range(6):
            if idx < n:
                positions[idx] = [0.08 + c * 0.18, 0.08 + r * 0.22]
                idx += 1
                
    # Add small random noise to break symmetry and avoid local traps
    np.random.seed(42)
    positions += np.random.uniform(-0.02, 0.02, positions.shape)
    positions = np.clip(positions, 0.05, 0.95)
    
    # Phase 1: Optimize positions to maximize minimum clearance
    res = minimize(_objective, positions.flatten(), method='Nelder-Mead', 
                   options={'maxiter': 40000, 'xatol': 1e-6, 'fatol': 1e-6})
    best_pos = res.x.reshape(n, 2)
    
    # Phase 2: Solve LP for optimal radii
    radii = _solve_radii(best_pos, n)
    
    return best_pos, radii, np.sum(radii)

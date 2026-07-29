# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 33c0c451) state=91d6f1d3 sum of radii=2.622318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N = 26
I_UPPER, J_UPPER = np.triu_indices(N, k=1)

def objective(x):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints:
    - Pairwise distance >= sum of radii
    - Circle boundaries within [0,1]x[0,1]
    Returns array of constraint values (must be >= 0)
    """
    cx, cy, r = x[0::3], x[1::3], x[2::3]
    
    # Vectorized pairwise distance constraints
    dx = cx[I_UPPER] - cx[J_UPPER]
    dy = cy[I_UPPER] - cy[J_UPPER]
    dists = np.hypot(dx, dy)
    c_dist = dists - (r[I_UPPER] + r[J_UPPER])
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    return np.concatenate([c_dist, c_bound])

def check_feasibility(x, tol=1e-7):
    """Check if a configuration satisfies constraints within tolerance"""
    c = constraints(x)
    return np.all(c >= -tol)

def create_hex_init(seed, scale=0.09, noise_scale=0.01):
    """Create a hexagonal lattice initialization with small feasible radii"""
    np.random.seed(seed)
    pts = []
    dy = np.sqrt(3) * scale
    dx = 2 * scale
    y = scale + 0.05
    col = 0
    while len(pts) < N:
        x = scale + 0.05 + (col % 2) * dx / 2
        while x <= 1.0 - scale - 0.05 and len(pts) < N:
            pts.append([x, y])
            x += dx
        y += dy
        col += 1
    pts = np.array(pts[:N])
    pts += np.random.randn(N, 2) * noise_scale
    pts = np.clip(pts, 0.02, 0.98)
    
    # Start with small uniform radii to guarantee initial feasibility
    r = np.full(N, 0.03)
    return np.concatenate([pts.flatten(), r])

def create_random_init(seed):
    """Create a random initialization in the safe inner region"""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    r = np.full(N, 0.03)
    return np.concatenate([pts.flatten(), r])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    best_sum = 0.0
    best_x = None
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Multiple restarts with mixed strategies
    for trial in range(60):
        if trial % 3 == 0:
            # Hexagonal grid with varying scales
            x0 = create_hex_init(trial, scale=0.08 + (trial % 5) * 0.01, noise_scale=0.005)
        elif trial % 3 == 1:
            # Random placement
            x0 = create_random_init(trial)
        else:
            # Perturb best found so far to escape local minima
            if best_x is not None:
                x0 = best_x + np.random.randn(3 * N) * 0.005
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            else:
                x0 = create_hex_init(trial)
                
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
        )
        
        # Accept if feasible and improves score
        if check_feasibility(res.x, tol=1e-6):
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
                
    # Fallback (highly unlikely to be reached)
    if best_x is None:
        best_x = create_hex_init(0)
        
    cx = best_x[0::3]
    cy = best_x[1::3]
    r = np.maximum(best_x[2::3], 0.0)
    
    # Clamp radii to strict boundary limits
    for i in range(N):
        r[i] = min(r[i], cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        
    # Iteratively fix any numerical overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < r[i] + r[j] - 1e-12:
                    overlap = r[i] + r[j] - d
                    r[i] -= overlap / 2.0
                    r[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    r = np.maximum(r, 0.0)
    centers = np.column_stack([cx, cy])
    return centers, r, float(np.sum(r))

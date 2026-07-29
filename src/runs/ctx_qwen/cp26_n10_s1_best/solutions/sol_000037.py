# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000021 (state 2060a481) state=80ac4c24 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
NUM_VARS = 3 * N

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def compute_constraints(x):
    """
    Computes all inequality constraints g(x) >= 0.
    Returns a 1D numpy array of constraint values.
    """
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    n_cons = 4 * N + N * (N - 1) // 2
    cons = np.empty(n_cons)
    idx = 0
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    for i in range(N):
        cons[idx] = cx[i] - cr[i]
        cons[idx+1] = 1.0 - cx[i] - cr[i]
        cons[idx+2] = cy[i] - cr[i]
        cons[idx+3] = 1.0 - cy[i] - cr[i]
        idx += 4
        
    # Overlap constraints: dist^2 >= (r_i + r_j)^2  =>  dx^2 + dy^2 - dr^2 >= 0
    for i in range(N):
        xi, yi, ri = cx[i], cy[i], cr[i]
        for j in range(i + 1, N):
            dx = xi - cx[j]
            dy = yi - cy[j]
            dr = ri + cr[j]
            cons[idx] = dx*dx + dy*dy - dr*dr
            idx += 1
    return cons

def get_bounds():
    """Returns variable bounds for [x, y, r] for each circle."""
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (radius cannot exceed half the square side)
    return bounds

def generate_initial_guess(seed):
    """Generates a feasible hexagonal lattice initialization with perturbation."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.085)
    
    r0 = 0.09
    idx = 0
    y = r0
    row = 0
    
    # Fill hexagonal grid until we have N points
    while idx < N and y + r0 < 1.0:
        x = r0
        # Shift odd rows by half-spacing for hexagonal packing
        if row % 2 == 1:
            x = r0 + r0 / 2.0
            
        while idx < N and x + r0 < 1.0:
            centers[idx, 0] = x + rng.uniform(-0.005, 0.005)
            centers[idx, 1] = y + rng.uniform(-0.005, 0.005)
            idx += 1
            x += 2 * r0
        y += r0 * np.sqrt(3)
        row += 1
        
    # Fallback if grid didn't yield enough points (unlikely with these params)
    while idx < N:
        centers[idx] = rng.uniform(0.15, 0.85, 2)
        radii[idx] = 0.05
        idx += 1
        
    # Flatten to optimization vector
    x0 = np.zeros(NUM_VARS)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_val = -np.inf
    best_x = None
    
    # Multi-start optimization with different random perturbations
    for seed in range(30):
        x0 = generate_initial_guess(seed)
        # Add small random noise to radii to break symmetry and help optimizer
        x0[2::3] += np.random.uniform(-0.005, 0.005, N)
        x0[2::3] = np.clip(x0[2::3], 0.05, 0.2)
        
        try:
            res = minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False}
            )
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            continue
            
    # Final refinement step on the best found configuration
    if best_x is not None:
        x0_refine = best_x + np.random.normal(0, 1e-4, NUM_VARS)
        try:
            res = minimize(
                objective, x0_refine, method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False}
            )
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Safety fallback if optimization completely failed
    if best_x is None:
        best_x = generate_initial_guess(0)
        best_val = 0.0

    # Extract results
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(N):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    return centers, radii, float(best_val)

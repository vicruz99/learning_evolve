# sol_000232 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2fe8b400) state=128bbfa3 sum of radii=2.546926 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Constants
N_CIRCLES = 26

def generate_initial_centers(n, seed=None):
    """
    Generates initial centers for n circles using a hexagonal lattice pattern.
    """
    if seed is not None:
        np.random.seed(seed)
    
    centers = np.zeros((n, 2))
    
    # Try to fill rows in a hexagonal pattern
    # Estimate spacing. For n=26, a 5x5 grid is close, but hexagonal is better.
    # Let's try to fit 5 circles per row.
    # Width available 1. 5 circles diameter 2r. 5*2r <= 1 => r <= 0.1.
    # Spacing dx = 0.2.
    # Vertical spacing dy = sqrt(3)/2 * 2r = 0.1732.
    
    # We need to place n circles.
    # Let's try to determine rows and cols.
    # Approx sqrt(n) ~ 5.
    
    # Let's just place them in a grid first and perturb for hex
    # Or simpler: random initialization with small radius is safer for optimizer start
    # But a structured start helps.
    
    # Let's create a grid of points and pick n points.
    # Grid 10x10
    points = []
    for i in range(10):
        for j in range(10):
            x = 0.05 + i * 0.09
            y = 0.05 + j * 0.09
            if x <= 1.0 and y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= n:
        # Shuffle and pick n
        idx = np.random.choice(len(points), n, replace=False)
        centers = points[idx]
    else:
        # Fallback to random
        centers = np.random.rand(n, 2)
        
    return centers

def objective_func(x):
    """
    Objective function: Minimize negative sum of radii.
    x is a 1D array of length 3*n: [x1, y1, r1, x2, y2, r2, ...]
    """
    n = len(x) // 3
    radii = x[2::3] # r1, r2, ...
    return -np.sum(radii)

def boundary_constraints(x):
    """
    Returns array of boundary constraint violations (must be >= 0).
    Constraints:
    x_i - r_i >= 0
    1 - x_i - r_i >= 0
    y_i - r_i >= 0
    1 - y_i - r_i >= 0
    """
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # We need to return a 1D array of values that must be >= 0
    c1 = xs - rs
    c2 = 1.0 - xs - rs
    c3 = ys - rs
    c4 = 1.0 - ys - rs
    
    return np.concatenate([c1, c2, c3, c4])

def overlap_constraints(x):
    """
    Returns array of overlap constraint violations (must be >= 0).
    Constraint: dist(i, j) - (r_i + r_j) >= 0
    Vectorized computation.
    """
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Centers matrix (n, 2)
    centers = np.column_stack((xs, ys))
    
    # Difference matrix (n, n, 2)
    # Broadcasting: centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # This creates a 3D array.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared distances
    dists_sq = np.sum(diff**2, axis=2)
    
    # Handle numerical issues where dists_sq might be slightly negative due to float error?
    # Unlikely with this setup, but sqrt needs non-negative.
    dists_sq = np.maximum(dists_sq, 0)
    dists = np.sqrt(dists_sq)
    
    # Radii sum matrix (n, n)
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    
    # Violations: dist - r_sum. We want dist >= r_sum.
    # So constraint value is dist - r_sum.
    constraint_vals = dists - r_sum
    
    # We only need upper triangular part (i < j)
    # Mask for upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    return constraint_vals[mask]

def run_packing():
    # Define bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (radius cannot be more than 0.5 in unit square)
    # Actually r can be bounded tighter, but 0.5 is safe.
    bounds = [(0, 1)] * (2 * N_CIRCLES) + [(0, 0.5)] * N_CIRCLES
    
    # Initial guess
    # Try a few different seeds to find a good local optimum
    best_result = None
    best_obj_val = np.inf
    
    # We will run optimization with a few random restarts
    num_restarts = 5
    
    for seed in range(num_restarts):
        # Generate initial centers
        initial_centers = generate_initial_centers(N_CIRCLES, seed=seed)
        initial_radii = np.ones(N_CIRCLES) * 0.05 # Small radius to ensure feasibility
        
        # Flatten to 1D vector
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = initial_centers[:, 0]
        x0[1::3] = initial_centers[:, 1]
        x0[2::3] = initial_radii
        
        # Constraints
        cons = [
            {'type': 'ineq', 'fun': boundary_constraints},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        # Optimization options
        # SLSQP is good for constrained problems
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
            
            if res.success and res.fun < best_obj_val:
                best_obj_val = res.fun
                best_result = res
        except Exception as e:
            print(f"Optimization failed for seed {seed}: {e}")
            continue
            
        # Also check if the result is feasible (just in case SLSQP didn't strictly enforce)
        # The validate_packing function will do this later, but good to be sure.
        # SLSQP usually returns a feasible point if one exists and maxiter is high enough.
        
    if best_result is None:
        # Fallback to a simple grid if optimization failed completely
        centers = np.zeros((N_CIRCLES, 2))
        radii = np.ones(N_CIRCLES) * 0.05
        # Place in grid
        idx = 0
        for r in range(6):
            for c in range(5):
                if idx < N_CIRCLES:
                    centers[idx] = [0.1 + c * 0.15, 0.1 + r * 0.15]
                    idx += 1
        return centers, radii, np.sum(radii)

    # Extract solution
    x_opt = best_result.x
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    
    for i in range(N_CIRCLES):
        centers[i, 0] = x_opt[3*i]
        centers[i, 1] = x_opt[3*i + 1]
        radii[i] = x_opt[3*i + 2]
        
    # Ensure non-negative radii (clipping just in case)
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# sol_000147 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 08bd70cf) state=46450b8f sum of radii=2.411698 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import itertools

def compute_max_radii(centers):
    """
    Solves the LP to maximize the sum of radii given fixed centers.
    """
    n = centers.shape[0]
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    c = -np.ones(n)

    # Constraints matrix A_ub * r <= b_ub
    constraints = []
    bounds_val = []

    # 1. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints.append(row)
        bounds_val.append(centers[i, 0])

        # r_i <= 1 - x_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints.append(row)
        bounds_val.append(1.0 - centers[i, 0])

        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints.append(row)
        bounds_val.append(centers[i, 1])

        # r_i <= 1 - y_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints.append(row)
        bounds_val.append(1.0 - centers[i, 1])

    # 2. Pairwise constraints: r_i + r_j <= distance(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints.append(row)
            bounds_val.append(dist)

    A_ub = np.array(constraints)
    b_ub = np.array(bounds_val)
    
    # Bounds for radii: r_i >= 0
    bounds_r = [(0, None) for _ in range(n)]

    try:
        # Use high-performance method; 'highs' is standard in modern scipy
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            return 0.0, np.zeros(n)
    except Exception:
        return 0.0, np.zeros(n)

def objective_wrapper(x_flat):
    """
    Wrapper for scipy optimizer.
    x_flat is the flattened centers array.
    Returns negative sum of radii (for minimization).
    """
    n = 26
    centers = x_flat.reshape(n, 2)
    # Clamp centers to [0, 1] to prevent invalid distances or boundary issues
    centers = np.clip(centers, 1e-5, 1 - 1e-5)
    sum_radii, radii = compute_max_radii(centers)
    return -sum_radii

def generate_hexagonal_init():
    """Generates a hexagonal packing initialization."""
    centers = []
    r_est = 0.09
    rows = 6
    cols = 5
    y_step = np.sqrt(3) * r_est
    
    y = r_est
    for row in range(rows):
        x = r_est
        num_cols = cols if row % 2 == 0 else cols - 1
        for col in range(num_cols):
            if len(centers) < 26:
                centers.append([x, y])
                x += 2 * r_est
        y += y_step
    centers = np.array(centers)
    return centers

def generate_perturbed_grid():
    """Generates a 5x5 grid plus one extra, perturbed."""
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    # Add 26th in center of a hole or random
    centers.append([0.5, 0.5])
    centers = np.array(centers)
    # Perturb
    centers += np.random.normal(0, 0.02, centers.shape)
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n_circles = 26
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # List of initial configurations to try
    inits = []
    
    # 1. Hexagonal packing
    inits.append(generate_hexagonal_init())
    
    # 2. Perturbed Grid
    inits.append(generate_perturbed_grid())
    
    # 3. Random dense packing
    for _ in range(3):
        centers = np.random.uniform(0.1, 0.9, (n_circles, 2))
        inits.append(centers)
        
    # 4. Corner-packed initialization
    centers_corner = []
    # 4 corners
    for x in [0.15, 0.85]:
        for y in [0.15, 0.85]:
            centers_corner.append([x, y])
    # Fill rest randomly
    while len(centers_corner) < n_circles:
        centers_corner.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    inits.append(np.array(centers_corner))

    for init_centers in inits:
        x0 = init_centers.flatten()
        
        # Local optimization to maximize sum of radii
        # Nelder-Mead is good for non-smooth objectives
        try:
            res = minimize(objective_wrapper, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500})
            current_centers = res.x.reshape(n_circles, 2)
            
            # Ensure centers are strictly within (0, 1) for validation safety
            current_centers = np.clip(current_centers, 1e-5, 1 - 1e-5)
            
            sum_rad, rad = compute_max_radii(current_centers)
            
            if sum_rad > best_sum:
                best_sum = sum_rad
                best_centers = current_centers.copy()
                best_radii = rad.copy()
        except Exception as e:
            continue

    # Final validation check (internal)
    # The LP guarantees constraints, but we double-check numerical stability
    if best_centers is not None:
        # Ensure no NaNs
        if np.isnan(best_centers).any() or np.isnan(best_radii).any():
            return np.zeros((26, 2)), np.zeros(26), 0.0
            
    return best_centers, best_radii, best_sum

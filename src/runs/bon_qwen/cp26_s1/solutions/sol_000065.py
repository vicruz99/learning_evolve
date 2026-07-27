# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cae61cda) state=95a597bf sum of radii=2.436482 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def solve_radii(centers):
    """
    Solves the Linear Programming problem to find optimal radii for a fixed set of centers.
    Maximizes sum(radii) subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Inequality Constraints: A_ub @ r <= b_ub
    A_rows = []
    b_vals = []
    
    # 1. Pairwise non-overlap constraints: r_i + r_j <= distance(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < 1e-12:
                dist = 1e-12 # Avoid zero distance singularity
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(dist)
            
    # 2. Boundary constraints: r_i <= distance to walls
    for i in range(n):
        x, y = centers[i]
        # Distance to left, right, bottom, top walls
        dists = [x, 1.0 - x, y, 1.0 - y]
        for d in dists:
            if d < 1e-12:
                d = 1e-12
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(d)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: radii must be non-negative
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        return np.zeros(n)

def objective_function(centers_flat):
    """
    Objective function for the optimizer.
    Takes flattened centers, reshapes, solves for radii, and returns negative sum of radii.
    """
    n = 26
    centers = centers_flat.reshape((n, 2))
    
    # Enforce bounds [0, 1] by clipping (though optimizer should handle this if we start inside)
    centers = np.clip(centers, 0, 1)
    
    radii = solve_radii(centers)
    return -np.sum(radii)

def generate_hex_grid(n, width=1.0, height=1.0):
    """
    Generates a hexagonal grid of points within [0, width] x [0, height].
    """
    points = []
    # Estimate rows and cols
    # Area per circle in hex packing ~ sqrt(3)/2 * r^2? No, area per point in lattice.
    # Spacing s. Area per point = s^2 * sqrt(3)/2.
    # We want n points. n * s^2 * sqrt(3)/2 approx width * height.
    # s approx sqrt(2 * area / (n * sqrt(3)))
    # But we just want a grid that fits.
    
    # Let's try to fit points with a specific spacing and trim
    spacing = 0.2 # Initial guess
    y = spacing / 2 # Start offset
    row = 0
    while y < height + spacing:
        x_start = spacing / 2 if row % 2 == 1 else 0
        x = x_start
        while x < width + spacing:
            points.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
        
    # If we have too many, trim from end. If too few, reduce spacing?
    # Actually, just returning a subset is fine for initialization.
    # But we want exactly n points.
    
    # Better approach: Distribute n points roughly evenly in hex pattern
    # Sort points by distance from center (0.5, 0.5)
    center = np.array([0.5, 0.5])
    points_arr = np.array(points)
    dists = np.linalg.norm(points_arr - center, axis=1)
    indices = np.argsort(dists)
    
    selected = points_arr[indices[:n]]
    return selected

def run_packing():
    n = 26
    initial_centers = generate_hex_grid(n)
    
    # Add small random noise to break symmetry and help optimizer
    np.random.seed(42)
    initial_centers += np.random.normal(0, 0.01, initial_centers.shape)
    initial_centers = np.clip(initial_centers, 0.01, 0.99)
    
    # Optimize centers
    # Flatten centers for optimizer
    x0 = initial_centers.flatten()
    
    # Use Nelder-Mead for derivative-free optimization
    res = minimize(objective_function, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000, 'adaptive': True})
    
    best_centers = res.x.reshape((n, 2))
    best_radii = solve_radii(best_centers)
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, float(sum_radii)

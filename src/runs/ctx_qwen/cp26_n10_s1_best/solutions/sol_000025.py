# sol_000025 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 764eb384) state=2d8d3e6f sum of radii=2.548420 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def generate_hexagonal_initial_guess(n, width=1.0, height=1.0):
    """
    Generates an initial configuration of n circles in a hexagonal pattern
    scaled and centered to fit roughly inside a width x height rectangle.
    """
    # Estimate rows
    # For hex packing, area ~ n * pi * r^2. Density ~ 0.9.
    # 1.0 * 1.0 * 0.9 ~ n * pi * r^2 => r ~ sqrt(0.9 / (n * pi))
    # But we just need a geometric layout.
    
    # Try to arrange in roughly triangular grid
    # Number of items in row k (0-indexed) in a triangle is k+1?
    # Let's just create a dense hexagonal grid and pick the first n points.
    
    # Approximate spacing based on packing density
    # If we pack n circles in unit square, avg radius ~ 0.1. Spacing ~ 0.2.
    spacing = 0.2
    
    points = []
    
    # Generate points in a hexagonal lattice
    # y increases by sqrt(3)/2 * spacing
    # x increases by spacing, shifted by spacing/2 for odd rows
    
    rows = int(math.sqrt(n * 2 / math.sqrt(3))) + 2
    cols = int(math.sqrt(n)) + 2
    
    for r in range(rows):
        for c in range(cols):
            x = c * spacing
            y = r * spacing * math.sqrt(3) / 2
            if r % 2 == 1:
                x += spacing / 2
            points.append((x, y))
    
    # Trim to n points
    points = points[:n]
    
    # Center and scale to fit in unit square with some margin
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Shift to center
    points_shifted = [(p[0] - center_x, p[1] - center_y) for p in points]
    
    # Scale to fit inside [0,1]x[0,1] with a margin (e.g., radius 0.1)
    # Max extent
    max_extent_x = max(abs(p[0]) for p in points_shifted)
    max_extent_y = max(abs(p[1]) for p in points_shifted)
    max_extent = max(max_extent_x, max_extent_y)
    
    # We want to leave room for radius. Let's say we want final r ~ 0.1.
    # So points should be in [-0.1, 0.9] roughly? No, [r, 1-r].
    # Let's scale so that the bounding box of centers fits in [0.15, 0.85] initially?
    # Actually, just scale to fit in [0,1] and let optimizer adjust.
    
    scale = 0.8 / max_extent if max_extent > 0 else 1.0
    
    points_scaled = [(p[0] * scale + 0.5, p[1] * scale + 0.5) for p in points_shifted]
    
    return np.array(points_scaled)

def objective_function(variables, n):
    """
    Objective to minimize: -sum(radii)
    Variables: [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = variables[2::3]
    return -np.sum(radii)

def constraint_overlap(variables, n, i, j):
    """
    Constraint: dist(i, j) >= r_i + r_j
    => dist(i, j) - r_i - r_j >= 0
    """
    xi, yi, ri = variables[3*i], variables[3*i+1], variables[3*i+2]
    xj, yj, rj = variables[3*j], variables[3*j+1], variables[3*j+2]
    
    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
    return dist - ri - rj

def constraint_boundary(variables, n, i, side):
    """
    Constraints:
    x - r >= 0  => x >= r
    y - r >= 0
    1 - x - r >= 0 => x + r <= 1
    1 - y - r >= 0
    
    side 0: left (x - r >= 0)
    side 1: right (1 - x - r >= 0)
    side 2: bottom (y - r >= 0)
    side 3: top (1 - y - r >= 0)
    """
    xi, yi, ri = variables[3*i], variables[3*i+1], variables[3*i+2]
    
    if side == 0:
        return xi - ri
    elif side == 1:
        return 1.0 - xi - ri
    elif side == 2:
        return yi - ri
    elif side == 3:
        return 1.0 - yi - ri
    return 0.0

def run_packing():
    n = 26
    
    # 1. Initial Guess
    # Use a rotated hexagonal packing to break symmetry and avoid grid traps
    centers_init = generate_hexagonal_initial_guess(n)
    
    # Add a small rotation to the initial guess to help find better packing
    angle = math.radians(15) # Try rotating by 15 degrees
    c_x, c_y = 0.5, 0.5
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    
    rotated_centers = []
    for x, y in centers_init:
        dx, dy = x - c_x, y - c_y
        rx = dx * cos_a - dy * sin_a + c_x
        ry = dx * sin_a + dy * cos_a + c_y
        rotated_centers.append([rx, ry])
    
    # Initial radii: small value to start valid
    r_init = 0.05
    variables_init = []
    for i in range(n):
        variables_init.extend([rotated_centers[i][0], rotated_centers[i][1], r_init])
    
    variables_init = np.array(variables_init)
    
    # 2. Constraints
    cons = []
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: constraint_overlap(v, n, i, j)
            })
            
    # Boundary constraints
    for i in range(n):
        for side in range(4):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, side=side: constraint_boundary(v, n, i, side)
            })
            
    # 3. Optimization
    # Use SLSQP which handles constraints well
    # Bounds: x,y in [0,1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, None)) # r
    
    res = minimize(
        objective_function,
        variables_init,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if res.success:
        best_vars = res.x
    else:
        # Fallback to initial guess optimized locally or just return init?
        # If failed, try to extract best from init?
        # But minimize usually returns best found.
        best_vars = res.x

    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i] = [best_vars[3*i], best_vars[3*i+1]]
        radii[i] = best_vars[3*i+2]
        
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To run locally to check
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Validation check would go here

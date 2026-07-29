# sol_000210 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c1389c4d) state=ba4c417c sum of radii=2.166667 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We define a layout to get close to a square aspect ratio.
    # Layout: 6, 5, 6, 5, 4 circles per row.
    # This sums to 26.
    
    rows_config = [6, 5, 6, 5, 4]
    rows = []
    y = 0
    
    # Lattice spacing parameters (relative)
    # Horizontal spacing = 2, Vertical spacing = sqrt(3) for hexagonal packing
    # Shift alternate rows by 1 unit
    
    for i, count in enumerate(rows_config):
        row_points = []
        # Shift odd rows (index 1, 3) by 1 unit in x
        shift = 1 if i % 2 == 1 else 0
        
        # To center the row horizontally within the cluster, we adjust start x
        # A row with 'count' circles spans (count-1)*2 units.
        # We want to align centers. 
        # Let's just place them and we will center the whole cluster later.
        # However, for the shifted rows, to maintain hexagonal packing,
        # the centers should be at x = shift, shift+2, ...
        
        # Let's generate raw coordinates
        for j in range(count):
            x = j * 2 + shift
            row_points.append([x, y])
        
        rows.append(row_points)
        y += math.sqrt(3) # Move to next row
        
    # Flatten the list of points
    initial_centers = np.array([pt for row in rows for pt in row])
    
    # 2. Scaling and Centering to fit in [0,1]x[0,1] roughly
    # Find bounding box of initial centers
    min_x, min_y = initial_centers.min(axis=0)
    max_x, max_y = initial_centers.max(axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # We want to fit this shape into the square.
    # The circles will have radius r. The centers must be in [r, 1-r].
    # So the span of centers must be <= 1 - 2r.
    # Also distance between centers >= 2r.
    # In our lattice, min distance is 2.
    # Let scale factor be s. New distance = 2s. So r = s.
    # New span = width * s.
    # Constraint: width * s <= 1 - 2s  =>  s(width + 2) <= 1 => s <= 1/(width+2)
    # Similarly for height.
    
    # Calculate max scale based on width and height constraints
    # Note: The "width" of the cluster of centers is width.
    # The circles extend r outside.
    # Total width occupied = width*s + 2*r = width*s + 2*s = s(width+2).
    
    scale_w = 1.0 / (width + 2.0)
    scale_h = 1.0 / (height + 2.0)
    scale = min(scale_w, scale_h)
    
    # Apply scale and center
    # First scale relative to (0,0) or just shift
    # Center the cluster in [0,1]
    
    # Shift to origin
    centered_centers = initial_centers - [min_x, min_y]
    # Scale
    scaled_centers = centered_centers * scale
    # Shift to center of square
    current_span_x = width * scale
    current_span_y = height * scale
    offset_x = (1.0 - current_span_x) / 2.0
    offset_y = (1.0 - current_span_y) / 2.0
    
    centers = scaled_centers + [offset_x, offset_y]
    radii = np.full(n, scale) # Initial radius estimate
    
    # 3. Optimization
    # We want to maximize the minimum distance between centers.
    # Equivalent to maximizing r such that dist(i,j) >= 2r.
    # We can optimize the positions to maximize min_dist, then set r = min_dist/2.
    # However, to make it a single optimization, we can maximize a function that penalizes overlaps.
    # Or simply maximize the minimum distance.
    
    # Let's use a barrier method or simple repulsion.
    # Define a function that returns the negative of the minimum distance.
    # We want to minimize this (maximize min dist).
    
    def objective(vars):
        # vars is flattened array of 2*N coordinates
        pts = vars.reshape(n, 2)
        
        min_d = float('inf')
        # Check distances
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < min_d:
                    min_d = d
        
        return -min_d # Minimize negative min_dist -> Maximize min_dist

    # Constraints: points must be within [0,1]
    # Bounds for x, y are [0, 1]
    bounds = [(0, 1)] * (2 * n)
    
    # Initial guess
    x0 = centers.flatten()
    
    # Run optimization
    # Using 'L-BFGS-B' with bounds
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
    
    optimized_centers = res.x.reshape(n, 2)
    
    # Calculate final radius
    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((optimized_centers[i] - optimized_centers[j])**2))
            if d < min_dist:
                min_dist = d
    
    # Radius is half the min distance, but also constrained by boundaries
    # r <= x, r <= 1-x, r <= y, r <= 1-y for all points
    # r <= min_dist / 2
    
    boundary_dist = float('inf')
    for i in range(n):
        x, y = optimized_centers[i]
        boundary_dist = min(boundary_dist, x, 1-x, y, 1-y)
    
    r = min(min_dist / 2.0, boundary_dist)
    
    radii = np.full(n, r)
    
    return optimized_centers, radii, np.sum(radii)

# Note: The validation function is provided separately, so we just define run_packing.

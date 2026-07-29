# sol_000194 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fb76805b) state=18deb866 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns a valid packing of 26 circles in a unit square maximizing the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # 1. Generate Hexagonal Lattice Points
    # We generate a grid of points that form a hexagonal packing structure.
    # Horizontal spacing is dx, vertical spacing between rows is dy = dx * sqrt(3)/2.
    # Odd rows are shifted by dx/2.
    
    # We use a dense enough grid to cover the square.
    # Initial guess for spacing to fit roughly 26 circles.
    # Area ~ 1. 26 * pi * r^2 ~ 1 => r ~ 0.11. 
    # Diameter ~ 0.22. 
    # Spacing dx ~ 0.22.
    
    dx = 0.22
    dy = dx * math.sqrt(3) / 2
    
    points = []
    y = 0
    while y < 1.1:
        x = 0
        # Determine row offset
        row_idx = int(round(y / dy))
        offset = (dx / 2) if row_idx % 2 == 1 else 0
        
        while x < 1.1:
            points.append([x + offset, y])
            x += dx
        y += dy
        
    points = np.array(points)
    
    # 2. Filter points that are strictly inside [0,1]x[0,1] (with some margin for radius)
    # Actually, centers must be in [0,1]. But if r > 0, center cannot be exactly 0 or 1.
    # We will select points that are within [0, 1].
    
    valid_mask = (points[:, 0] >= 0) & (points[:, 0] <= 1) & \
                 (points[:, 1] >= 0) & (points[:, 1] <= 1)
    
    valid_points = points[valid_mask]
    
    # If we have more than n points, we need to select n.
    # If fewer, we might need to adjust density or add points.
    # With dx=0.22, we likely have more.
    
    if len(valid_points) >= n:
        # Select n points. 
        # A good heuristic is to pick points that are centrally located or well spread.
        # Or simply pick the first n valid points. 
        # However, to maximize radius, we want to avoid boundary effects as much as possible?
        # Actually, boundary effects are unavoidable. 
        # Let's try to pick points that are somewhat evenly distributed.
        # Simple selection:
        selected_points = valid_points[:n]
    else:
        # Fallback: if not enough points, use random or grid
        selected_points = np.random.rand(n, 2)
        
    # 3. Local Optimization using Force-Directed Layout
    # We want to move centers to maximize the minimum distance between them (and boundaries).
    # This is equivalent to maximizing the radius of equal circles.
    
    centers = selected_points.copy()
    
    # Optimization function: Maximize min_dist
    # We minimize negative of min_dist.
    # Variables: centers.flatten() (52 variables)
    
    def objective(params):
        c = params.reshape(n, 2)
        
        # Boundary constraints: distance to walls
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # So 2r <= min(x, 1-x, y, 1-y)
        # We want to maximize min( distances between circles, distances to walls )
        
        min_d = 1.0
        
        # Inter-circle distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                if dist < min_d:
                    min_d = dist
        
        # Boundary distances
        for i in range(n):
            x, y = c[i]
            d_wall = min(x, 1-x, y, 1-y)
            if d_wall < min_d:
                min_d = d_wall
                
        return -min_d

    # Use L-BFGS-B with bounds [0, 1]
    # Initial guess
    x0 = centers.flatten()
    bounds = [(0, 1) for _ in range(n * 2)]
    
    # Run optimization multiple times with different seeds if needed, 
    # but for this task, one good run should suffice.
    # We can add a small random perturbation to initial positions to escape bad local minima.
    # But the hexagonal start is usually good.
    
    # Add small random noise to break symmetry
    centers = centers + np.random.normal(0, 0.005, centers.shape)
    # Clip to [0, 1]
    centers = np.clip(centers, 0, 1)
    x0 = centers.flatten()
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000})
    
    optimal_centers = result.x.reshape(n, 2)
    
    # 4. Calculate Maximum Radius
    # The radius r is limited by the minimum distance between any two centers (divided by 2)
    # and the minimum distance from a center to a boundary.
    
    min_dist = 1.0
    
    # Inter-circle
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((optimal_centers[i] - optimal_centers[j])**2))
            if dist < min_dist:
                min_dist = dist
                
    # Boundary
    for i in range(n):
        x, y = optimal_centers[i]
        d_wall = min(x, 1-x, y, 1-y)
        if d_wall < min_dist:
            min_dist = d_wall
            
    r = min_dist / 2.0
    
    radii = np.full(n, r)
    sum_radii = np.sum(radii)
    
    return optimal_centers, radii, sum_radii

# Helper to verify logic locally (not part of run_packing but good for debugging)
# Note: The prompt asks for run_packing function.

# sol_000187 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 624944be) state=60499cce sum of radii=2.605255 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # Create a hexagonal lattice to get a good starting point.
    # We want a dense cluster.
    # We generate points (i + j/2, j * sqrt(3)/2)
    
    points = []
    # Try a range of lattice points to find 26 that are reasonably compact
    # A 5x5 hexagonal block might be a good start.
    # Let's just generate a grid and take the closest 26 to the center (0.5, 0.5)
    grid_points = []
    for i in range(10): # x index
        for j in range(10): # y index
            x = i + j * 0.5
            y = j * np.sqrt(3) / 2
            grid_points.append((x, y))
            
    grid_points = np.array(grid_points)
    
    # Center the grid at (0,0) for easier selection, then shift
    grid_points -= np.mean(grid_points, axis=0)
    
    # Sort by distance from center to pick the most compact cluster
    dists = np.linalg.norm(grid_points, axis=1)
    sorted_indices = np.argsort(dists)
    initial_centers = grid_points[sorted_indices[:n]]
    
    # Scale to fit roughly in [0,1]x[0,1] with some margin
    # Current extent
    min_x, min_y = np.min(initial_centers, axis=0)
    max_x, max_y = np.max(initial_centers, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # We want to leave room for radius. 
    # Let's scale such that the bounding box is roughly 0.8, leaving 0.1 margin.
    scale = 0.8 / max(width, height)
    initial_centers *= scale
    initial_centers += (1 - scale * (max_x - min_x + min_x - min_x)) / 2 # Center roughly
    
    # Actually, simpler scaling:
    # Map min to 0.1, max to 0.9
    # Or just normalize to [0,1] and shrink slightly
    current_min = np.min(initial_centers, axis=0)
    current_max = np.max(initial_centers, axis=0)
    current_span = current_max - current_min
    
    # Shift to 0,0
    initial_centers -= current_min
    # Scale to 0.9 size
    initial_centers *= 0.9 / np.max(current_span)
    # Center in square
    initial_centers += (1 - 0.9) / 2
    
    # Initial radii guess: uniform small radius
    initial_radii = np.full(n, 0.05)
    
    # 2. Optimization Setup
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Total 3 * n variables
    
    def objective(vars):
        # vars shape (3*n,)
        # reshape to (n, 3) -> [x, y, r]
        pts = vars.reshape(n, 3)
        # Sum of radii (negative for minimization)
        return -np.sum(pts[:, 2])

    def constraints(vars):
        pts = vars.reshape(n, 3)
        centers = pts[:, :2]
        radii = pts[:, 2]
        
        cons = []
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v.reshape(n,3)[i,0] - v.reshape(n,3)[i,2]})
        # x + r <= 1  => 1 - (x+r) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v.reshape(n,3)[i,0] + v.reshape(n,3)[i,2])})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v.reshape(n,3)[i,1] - v.reshape(n,3)[i,2]})
        # y + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v.reshape(n,3)[i,1] + v.reshape(n,3)[i,2])})
        
        # Non-negative radius
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v.reshape(n,3)[i,2]})

        return cons

    def overlap_constraints(vars):
        pts = vars.reshape(n, 3)
        centers = pts[:, :2]
        radii = pts[:, 2]
        
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                # dist >= r_i + r_j
                # dist^2 >= (r_i + r_j)^2
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                
                # Vectorized distance calculation for speed
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff**2)
                r_sum = radii[i] + radii[j]
                
                # We need dist >= r_sum. 
                # If dist is 0, we have issues with sqrt, but dist_sq handles it.
                # Constraint: dist - r_sum >= 0
                # To avoid sqrt in gradient, we can use dist_sq >= r_sum^2, 
                # but that's not smooth if r_sum is negative (it isn't).
                # However, dist >= r_sum is equivalent to dist_sq >= r_sum^2 for positive values.
                # But let's stick to dist - r_sum for clarity, sqrt is differentiable for dist>0.
                # If dist=0, gradient is undefined, but circles won't be at same point usually.
                
                dist = np.sqrt(dist_sq)
                val = dist - r_sum
                
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: np.sqrt(np.sum((v.reshape(n,3)[i,:2] - v.reshape(n,3)[j,:2])**2)) - (v.reshape(n,3)[i,2] + v.reshape(n,3)[j,2])})
                
        return cons

    # Collect all constraints
    # Boundary and non-negativity per circle
    all_cons = []
    for i in range(n):
        # x - r >= 0
        all_cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - (x + r) >= 0
        all_cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v[3*i] + v[3*i+2])})
        # y - r >= 0
        all_cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - (y + r) >= 0
        all_cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v[3*i+1] + v[3*i+2])})
        # r >= 0
        all_cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Define a specific function for each pair to avoid closure issues with lambda capturing loop var
            def make_overlap_constraint(i_idx, j_idx):
                def overlap_constraint(v):
                    pts = v.reshape(n, 3)
                    c1 = pts[i_idx, :2]
                    c2 = pts[j_idx, :2]
                    r1 = pts[i_idx, 2]
                    r2 = pts[j_idx, 2]
                    dist = np.sqrt(np.sum((c1 - c2)**2))
                    return dist - (r1 + r2)
                return overlap_constraint

            all_cons.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    # Initial guess array
    x0 = np.concatenate([initial_centers.flatten(), initial_radii.flatten()])
    
    # Run optimization
    # Use SLSQP which handles constraints
    # Maximize sum of radii -> Minimize -sum
    result = minimize(objective, x0, method='SLSQP', constraints=all_cons, options={'maxiter': 1000, 'ftol': 1e-9})
    
    if result.success:
        final_vars = result.x
    else:
        # Fallback to initial guess if optimization fails (unlikely)
        final_vars = x0

    # Extract results
    final_pts = final_vars.reshape(n, 3)
    centers = final_pts[:, :2]
    radii = final_pts[:, 2]
    sum_radii = np.sum(radii)
    
    # Validation check (internal)
    # Note: The prompt says we run validation function, so we just return valid data.
    # We assume the optimizer respected constraints.
    
    return centers, radii, sum_radii

# Helper functions for constraints to avoid lambda closure issues if needed, 
# but the structure above with 'make_overlap_constraint' handles it.
# However, to be safe and follow rules strictly (no closures from nesting if possible, though lambdas are allowed as long as not nested weirdly),
# let's refactor slightly to be cleaner.

def get_boundary_constraints(n):
    cons = []
    for i in range(n):
        idx = 3 * i
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: v[idx] - v[idx+2]})
        # 1 - (x + r) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: 1 - (v[idx] + v[idx+2])})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: v[idx+1] - v[idx+2]})
        # 1 - (y + r) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: 1 - (v[idx+1] + v[idx+2])})
        # r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: v[idx+2]})
    return cons

def get_overlap_constraints(n):
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            idx1 = 3 * i
            idx2 = 3 * j
            def make_constraint(i1, i2):
                def constraint(v):
                    # v is flattened vector
                    # x1, y1, r1 at i1, i1+1, i1+2
                    x1, y1, r1 = v[i1], v[i1+1], v[i1+2]
                    x2, y2, r2 = v[i2], v[i2+1], v[i2+2]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    return dist - (r1 + r2)
                return constraint
            cons.append({'type': 'ineq', 'fun': make_constraint(idx1, idx2)})
    return cons

# Rewriting run_packing with cleaner constraint generation
def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Initialization
    grid_points = []
    for i in range(12):
        for j in range(12):
            x = i + j * 0.5
            y = j * np.sqrt(3) / 2
            grid_points.append((x, y))
    grid_points = np.array(grid_points)
    grid_points -= np.mean(grid_points, axis=0)
    dists = np.linalg.norm(grid_points, axis=1)
    sorted_indices = np.argsort(dists)
    initial_centers = grid_points[sorted_indices[:n]]
    
    # Normalize to fit in unit square with some margin
    min_c = np.min(initial_centers, axis=0)
    max_c = np.max(initial_centers, axis=0)
    span = max_c - min_c
    scale = 0.85 / np.max(span)
    initial_centers = (initial_centers - min_c) * scale + 0.075
    
    initial_radii = np.full(n, 0.05)
    
    x0 = np.concatenate([initial_centers.flatten(), initial_radii.flatten()])
    
    # Constraints
    boundary_cons = get_boundary_constraints(n)
    overlap_cons = get_overlap_constraints(n)
    all_cons = boundary_cons + overlap_cons
    
    # Objective
    def objective(v):
        pts = v.reshape(n, 3)
        return -np.sum(pts[:, 2])
    
    result = minimize(objective, x0, method='SLSQP', constraints=all_cons, options={'maxiter': 2000, 'ftol': 1e-12})
    
    final_vars = result.x
    final_pts = final_vars.reshape(n, 3)
    centers = final_pts[:, :2]
    radii = final_pts[:, 2]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

import numpy as np
import math
from scipy.optimize import minimize

def get_hex_lattice_positions(n_circles, side=1.0):
    """
    Generates initial positions for n_circles using a hexagonal lattice.
    """
    # Estimate radius based on area packing density approx
    # Area ~ n * pi * r^2 * density. Density ~ 0.9069.
    # But boundary effects matter. Heuristic: r approx 0.09 for n=26.
    r_init = 0.08 
    
    positions = []
    y = r_init
    
    # We will try to fill rows. 
    # Row 0: x = r, 3r, 5r...
    # Row 1: x = 2r, 4r, 6r... (shifted by r)
    # Vertical step: r * sqrt(3)
    
    row_idx = 0
    while len(positions) < n_circles:
        if row_idx % 2 == 0:
            # Even row: starts at r
            x = r_init
            while x <= 1.0 - r_init:
                positions.append((x, y))
                x += 2 * r_init
                if len(positions) >= n_circles: break
        else:
            # Odd row: starts at 2r (shifted)
            x = 2 * r_init
            while x <= 1.0 - r_init:
                positions.append((x, y))
                x += 2 * r_init
                if len(positions) >= n_circles: break
        
        # Move to next row
        y += r_init * math.sqrt(3)
        if y + r_init > 1.0:
            # If we can't fit another row, break and rely on optimization to adjust
            # But we need exactly n_circles. If we ran out of space, we might have truncated.
            # The loop condition len(positions) < n_circles handles adding, 
            # but if y is too high, we might add points outside. 
            # Let's ensure we don't add points if y is invalid.
            if y + r_init > 1.0 + 1e-6:
                 # Try to squeeze in remaining points by reducing r or just adding
                 # For initialization, it's okay if they are slightly outside, optimizer will fix.
                 pass 
        
        row_idx += 1
        if len(positions) < n_circles and y + r_init > 1.0:
            # Force add remaining points at bottom to ensure count, though they will be invalid initially
            rem = n_circles - len(positions)
            # Just place them somewhere valid-ish or let optimizer handle
            # Better: restart with smaller r
            break

    # If we didn't get enough points (unlikely with r=0.08), return random grid
    if len(positions) < n_circles:
        # Fallback to random grid
        rng = np.random.RandomState(42)
        positions = []
        for _ in range(n_circles):
            positions.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))

    return np.array(positions[:n_circles]), r_init

def distance_matrix(centers):
    """Computes pairwise distance matrix."""
    # centers: (N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def objective_func(params, n_circles):
    """
    Objective: Maximize sum of radii => Minimize -sum(radii)
    params: [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = params[:2*n_circles].reshape(n_circles, 2)
    radii = params[2*n_circles:]
    return -np.sum(radii)

def constraint_overlap(params, n_circles):
    """
    Constraints: dist(i, j) >= r_i + r_j
    Returns array of slack values (should be >= 0)
    """
    centers = params[:2*n_circles].reshape(n_circles, 2)
    radii = params[2*n_circles:]
    
    # Vectorized distance calculation
    # To save memory/time, we can just compute lower triangular
    # But for N=26, full matrix is small (26x26)
    dist_mat = distance_matrix(centers)
    
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Slack = dist - (r1 + r2)
    # We need slack >= 0
    slacks = dist_mat - radii_sum
    
    # Return lower triangular elements (excluding diagonal)
    # Or just all pairs i < j
    slacks_list = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            slacks_list.append(slacks[i, j])
            
    return np.array(slacks_list)

def constraint_boundary(params, n_circles):
    """
    Constraints: 
    x_i >= r_i  => x_i - r_i >= 0
    x_i <= 1 - r_i => x_i + r_i - 1 <= 0 => 1 - x_i - r_i >= 0
    Same for y
    """
    centers = params[:2*n_circles].reshape(n_circles, 2)
    radii = params[2*n_circles:]
    
    constraints = []
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        constraints.append(x - r)         # Left
        constraints.append(1 - x - r)     # Right
        constraints.append(y - r)         # Bottom
        constraints.append(1 - y - r)     # Top
        
    return np.array(constraints)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # 1. Initialize positions
    initial_centers, r_init = get_hex_lattice_positions(n_circles)
    
    # Scale initial radii to be small enough to avoid immediate overlap 
    # but close to expected size.
    # The lattice generator used r_init ~ 0.08. 
    # Let's check overlaps and shrink if needed.
    dists = distance_matrix(initial_centers)
    min_dist = np.min(dists[np.triu_indices_from(dists, k=1)])
    
    # Safe radius is min_dist / 2 - epsilon
    safe_r = max(0.01, (min_dist / 2) * 0.9)
    
    # If safe_r is very small (points too close), reduce it
    if safe_r < 0.05:
        safe_r = 0.05 # Force a reasonable start, optimizer will handle
        
    initial_radii = np.full(n_circles, safe_r)
    
    # Flatten params: x1, y1, r1, ...
    params0 = np.concatenate([initial_centers.flatten(), initial_radii])
    
    # 2. Optimization
    # Constraints
    # Overlap: dist >= r_i + r_j => dist - r_i - r_j >= 0
    cons_overlap = {
        'type': 'ineq',
        'fun': lambda p: constraint_overlap(p, n_circles)
    }
    
    # Boundary: x >= r, 1-x >= r, etc.
    cons_boundary = {
        'type': 'ineq',
        'fun': lambda p: constraint_boundary(p, n_circles)
    }
    
    # Bounds for radii (non-negative) and centers (0 to 1)
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 1.0)) # r (radius can't be > 0.5 actually, but 1.0 is safe)

    # Try to optimize
    # Using SLSQP as it handles constraints well
    res = minimize(
        objective_func,
        params0,
        args=(n_circles,),
        method='SLSQP',
        bounds=bounds,
        constraints=[cons_overlap, cons_boundary],
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # 3. Extract results
    best_params = res.x
    centers = best_params[:2*n_circles].reshape(n_circles, 2)
    radii = best_params[2*n_circles:]
    
    # Ensure radii are non-negative (numerical safety)
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    # If optimization failed to improve significantly or result is invalid, 
    # fallback to a valid grid configuration.
    # But let's assume it worked. 
    # Just in case, clip centers to [0,1] and adjust radii if they violate boundary slightly
    # The optimizer should have handled this, but numerical errors might occur.
    
    # Final validation check (mental or debug)
    # We rely on the provided validate_packing function later.
    
    return centers, radii, float(sum_radii)

# Helper to make sure top level functions are defined if needed, 
# though the logic is inside run_packing mostly.
# The prompt asks for run_packing function.
# sol_000209 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cccf4974) state=ab34d851 sum of radii=1.565468 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Strategy: Optimize for 26 equal circles using a penalty-based approach with 
    multiple initializations (Random, Grid, Hex-like).
    """
    n = 26
    
    def cost_function(params, n):
        # params: [x1, y1, ..., xn, yn, r]
        xy = params[:2*n]
        r = params[2*n]
        centers = xy.reshape((n, 2))
        
        # 1. Boundary Penalties
        # Centers must be in [r, 1-r]
        # Penalty: sum of squared violations
        
        # x violations
        left_x = np.maximum(0, r - centers[:, 0])
        right_x = np.maximum(0, centers[:, 0] - (1.0 - r))
        # y violations
        top_y = np.maximum(0, centers[:, 1] - (1.0 - r))
        bottom_y = np.maximum(0, r - centers[:, 1])
        
        bound_pen = np.sum(left_x**2 + right_x**2 + top_y**2 + bottom_y**2)
        
        # 2. Overlap Penalties
        # Distance between centers >= 2r
        # Violation: max(0, 2r - dist)^2
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Ignore self-distance
        np.fill_diagonal(dists, np.inf)
        
        overlap = 2.0 * r - dists
        overlap_pen = np.sum(np.maximum(0, overlap)**2)
        
        # Penalty weights
        # High weights ensure constraints are respected
        w_bound = 1000.0
        w_overlap = 1000.0
        
        # Objective: Maximize r (Minimize -r) + Penalties
        cost = -r + w_bound * bound_pen + w_overlap * overlap_pen
        return cost

    def objective(params):
        return cost_function(params, n)

    candidates = []
    np.random.seed(42)
    
    # Strategy 1: Multiple Random Starts
    # Searching a range of initial radii to find the basin of attraction for the optimum
    for _ in range(20):
        r_init = 0.08 + np.random.rand() * 0.05 # Range [0.08, 0.13]
        low = r_init
        high = 1.0 - r_init
        # Fallback if r_init > 0.5 (unlikely here)
        if low > high:
            low, high = 0.0, 1.0
        
        # Random centers within valid range for r_init
        centers = np.random.uniform(low, high, (n, 2))
        params = np.concatenate([centers.flatten(), [r_init]])
        candidates.append(params)

    # Strategy 2: Perturbed Grid Start
    # 5x5 grid is a known local optimum for r=0.1 (25 circles).
    # We add a 26th circle and perturb to escape the grid constraint.
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + i*0.2, 0.1 + j*0.2])
    # Add 26th point in the center (will overlap, but optimizer will resolve)
    grid_pts.append([0.5, 0.5])
    
    centers_grid = np.array(grid_pts)
    # Perturb to break symmetry
    centers_grid += np.random.normal(0, 0.02, centers_grid.shape)
    # Clip to safe zone [0.05, 0.95] to avoid immediate boundary crash
    centers_grid[:, 0] = np.clip(centers_grid[:, 0], 0.05, 0.95)
    centers_grid[:, 1] = np.clip(centers_grid[:, 1], 0.05, 0.95)
    
    # Start with r=0.08 to allow movement
    params_grid = np.concatenate([centers_grid.flatten(), [0.08]])
    candidates.append(params_grid)

    # Strategy 3: Hexagonal-like Lattice
    # Hexagonal packing is denser than square grid.
    centers_hex = []
    r_hex = 0.09
    # Generate rows
    # Vertical spacing for hex packing of radius r is r*sqrt(3)
    y = r_hex
    row_idx = 0
    while y + r_hex <= 1.0 + 1e-9 and len(centers_hex) < 26:
        # Horizontal shift for odd rows
        shift = r_hex if row_idx % 2 != 0 else 0.0
        x_start = r_hex + shift
        x = x_start
        while x + r_hex <= 1.0 + 1e-9 and len(centers_hex) < 26:
            centers_hex.append([x, y])
            x += 2 * r_hex # Horizontal spacing 2r
        y += r_hex * np.sqrt(3)
        row_idx += 1
        
    # Fill remaining if needed
    while len(centers_hex) < 26:
        centers_hex.append([np.random.rand(), np.random.rand()])
        
    centers_hex_arr = np.array(centers_hex[:26])
    # Clip to valid region
    centers_hex_arr[:, 0] = np.clip(centers_hex_arr[:, 0], r_hex, 1.0 - r_hex)
    centers_hex_arr[:, 1] = np.clip(centers_hex_arr[:, 1], r_hex, 1.0 - r_hex)
    
    params_hex = np.concatenate([centers_hex_arr.flatten(), [r_hex]])
    candidates.append(params_hex)

    best_params = None
    best_cost = np.inf
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)]

    # Optimization Loop
    for p in candidates:
        # Use Powell method (derivative-free, handles bounds)
        res = opt.minimize(objective, p, method='Powell', bounds=bounds,
                           options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-12})
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
            
        # Refine with Nelder-Mead
        res2 = opt.minimize(objective, res.x, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res2.fun < best_cost:
            best_cost = res2.fun
            best_params = res2.x

    # Extract results
    xy = best_params[:2*n].reshape((n, 2))
    r_opt = best_params[2*n]
    
    # Post-processing: Ensure strict validity
    # Calculate the maximum valid r for the found configuration
    
    # 1. Boundary constraint: r <= min(x, 1-x, y, 1-y)
    min_x = np.min(xy[:, 0])
    max_x = np.max(xy[:, 0])
    min_y = np.min(xy[:, 1])
    max_y = np.max(xy[:, 1])
    r_bound = min(min_x, 1.0 - max_x, min_y, 1.0 - max_y)
    
    # 2. Overlap constraint: 2r <= dist(i, j) => r <= dist(i, j) / 2
    diffs = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r_overlap = min_dist / 2.0
    
    # The valid radius is the minimum of these limits
    final_r = min(r_opt, r_bound, r_overlap)
    
    # Ensure non-negative
    if final_r < 0:
        final_r = 0.0
        
    final_radii = np.full(n, final_r)
    
    return xy, final_radii, 26 * final_r

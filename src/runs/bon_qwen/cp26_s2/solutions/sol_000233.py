# sol_000233 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b088ff81) state=723942e7 sum of radii=1.662198 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

# Global constant for number of circles
N = 26

def objective(v):
    """
    Objective function to minimize.
    We aim to maximize the minimum radius r feasible for the configuration of centers.
    The function returns -r, so minimizing the objective maximizes r.
    """
    # Reshape vector to (N, 2) array of centers
    coords = v.reshape(N, 2)
    
    # Check bounds [0, 1]
    # If coordinates are outside the unit square, return a large penalty
    if np.any(coords < 0.0) or np.any(coords > 1.0):
        return 1e6

    # Calculate distance of each circle center to the nearest boundary
    # The radius r is limited by min(x, 1-x, y, 1-y)
    dists_wall = np.minimum(np.minimum(coords[:, 0], 1.0 - coords[:, 0]), 
                            np.minimum(coords[:, 1], 1.0 - coords[:, 1]))
    min_wall = np.min(dists_wall)
    
    # Calculate pairwise distances between all circle centers
    # We use vectorized operations for efficiency
    # diff shape: (N, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Distance from a circle to itself is 0, set to infinity to ignore
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is constrained by half the minimum pairwise distance
    min_pair_r = min_pair_dist / 2.0
    
    # The feasible radius for this configuration is the minimum of wall constraints and pair constraints
    min_r = min(min_wall, min_pair_r)
    
    # Return negative radius to maximize it via minimization
    return -min_r

def get_initial_centers():
    """
    Generates an initial configuration based on a hexagonal grid pattern.
    This provides a good starting point for the optimizer.
    """
    # Configuration for 26 circles: 5 rows with counts 6, 5, 6, 5, 4 (Total 26)
    rows_config = [6, 5, 6, 5, 4]
    centers = []
    n_rows = len(rows_config)
    
    # Margins to keep away from edges initially
    y_margin = 0.1
    
    # Distribute rows vertically
    y_coords = np.linspace(y_margin, 1.0 - y_margin, n_rows)
    
    for r_idx, count in enumerate(rows_config):
        y = y_coords[r_idx]
        
        # Horizontal shift for hexagonal packing (odd rows shifted)
        shift = 0.0
        if r_idx % 2 == 1:
            shift = 0.08 
        
        # X range adjusted for shift
        x_start = y_margin + shift
        x_end = 1.0 - y_margin + shift
        
        if count == 1:
            x_vals = [0.5 + shift]
        else:
            x_vals = np.linspace(x_start, x_end, count)
        
        for x in x_vals:
            # Clip to ensure valid initial coordinates
            x = np.clip(x, 0.0, 1.0)
            centers.append([x, y])
    
    return np.array(centers)

def run_packing() -> tuple:
    # Generate initial centers
    init_centers = get_initial_centers()
    x0 = init_centers.flatten()
    
    # Bounds for coordinates [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_r = 0.0
    best_centers = None
    
    # Run optimization with Nelder-Mead
    # Nelder-Mead is derivative-free and handles non-smooth objectives reasonably well
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
        r_val = -res.fun
        if r_val > best_r:
            best_r = r_val
            best_centers = res.x.reshape(N, 2)
    except Exception:
        pass

    # Random restarts to escape local minima
    for _ in range(5):
        x0_rand = np.random.rand(2 * N) * 0.8 + 0.1
        try:
            res = opt.minimize(objective, x0_rand, method='Nelder-Mead', bounds=bounds, 
                               options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9})
            r_val = -res.fun
            if r_val > best_r:
                best_r = r_val
                best_centers = res.x.reshape(N, 2)
        except Exception:
            pass

    # Fallback if optimization failed
    if best_centers is None:
        best_centers = init_centers

    # Compute precise final radius for the best configuration found
    coords = best_centers
    min_pair_dist = np.inf
    min_wall = np.inf
    
    # Check wall distances
    for i in range(N):
        x, y = coords[i]
        d = min(x, 1.0 - x, y, 1.0 - y)
        if d < min_wall:
            min_wall = d
        
        # Check pairwise distances
        for j in range(i + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_pair_dist:
                min_pair_dist = d
    
    # The valid radius is the minimum of half the closest pair distance and closest wall distance
    r_final = min(min_pair_dist / 2.0, min_wall)
    if r_final < 0.0:
        r_final = 0.0
        
    # All circles have the same radius
    radii = np.full(N, r_final)
    centers = best_centers
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii

import numpy as np
import scipy.optimize as opt

def compute_overlap_penalty(centers, radii):
    """
    Compute a penalty score based on overlaps and boundary violations.
    Returns the total penalty.
    """
    n = len(radii)
    penalty = 0.0
    
    # Boundary violations
    # Circle i is valid if r_i <= x_i <= 1 - r_i and r_i <= y_i <= 1 - r_i
    # Penalty is squared distance beyond boundary
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left
        if x - r < 0: penalty += (x - r)**2
        # Right
        if x + r > 1: penalty += (x + r - 1)**2
        # Bottom
        if y - r < 0: penalty += (y - r)**2
        # Top
        if y + r > 1: penalty += (y + r - 1)**2

    # Pairwise overlaps
    # Vectorized computation for efficiency
    # dist^2 = (x_i - x_j)^2 + (y_i - y_j)^2
    # We want dist >= r_i + r_j
    # Penalty if r_i + r_j - dist > 0
    
    # Using broadcasting for all pairs
    # centers shape: (n, 2)
    # radii shape: (n,)
    
    # Compute pairwise squared distances
    # diff shape: (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid self-interaction and duplicates by masking
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Distances
    dist = np.sqrt(dist_sq + 1e-15) # epsilon to avoid sqrt(0) issues if needed, though mask handles diag
    
    # Required distances
    req_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap amount (positive means overlap)
    overlap = req_dist - dist
    
    # Only consider upper triangle
    overlap_masked = overlap[mask]
    
    # Penalty: sum of squared overlaps
    penalty += np.sum(np.maximum(overlap_masked, 0)**2)
    
    return penalty

def objective_func(vars_flat, n_circles):
    """
    Objective function for scipy.optimize.
    vars_flat contains [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    We want to maximize sum(r_i), so we minimize -sum(r_i) + penalty.
    """
    centers = vars_flat[:2*n_circles].reshape(-1, 2)
    radii = vars_flat[2*n_circles:]
    
    # Enforce non-negative radii implicitly by mapping if needed, 
    # but bounds will handle it. Just ensure no negative in penalty calc if it goes there.
    radii = np.maximum(radii, 0)
    
    # Objective: maximize sum of radii => minimize -sum(radii)
    obj = -np.sum(radii)
    
    # Add penalty for constraints
    # Weight for penalty needs to be high to enforce constraints
    penalty_weight = 100.0
    penalty = compute_overlap_penalty(centers, radii)
    
    return obj + penalty_weight * penalty

def get_initial_configs(n=26):
    """Generate a list of starting configurations."""
    configs = []
    
    # 1. Hexagonal lattice
    # Approximate radius for hex packing of 26 in square
    # We want a dense start. r=0.1 is too big for 26 in simple grid, but hex might fit.
    # Let's start with small r and expand, or just r=0.08
    r_start = 0.08
    centers = []
    row = 0
    while len(centers) < n:
        y = r_start + row * r_start * np.sqrt(3)
        if y + r_start > 1.0:
            break
        shift = (r_start if row % 2 == 1 else 0)
        x = r_start + shift
        while x + r_start <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_start
        row += 1
    
    # If hex didn't fill 26, add random ones in gaps or just truncate/pad
    while len(centers) < n:
        centers.append([0.5, 0.5]) # Placeholder
    
    centers = np.array(centers[:n])
    radii = np.full(n, r_start)
    configs.append((centers, radii))
    
    # 2. 5x5 Grid + 1 center (perturbed)
    # 5x5 fits r=0.1 exactly. 26th circle must be small or grid shrunk.
    # Start with r=0.09 to allow movement.
    r_grid = 0.09
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([r_grid + i * 2 * r_grid, r_grid + j * 2 * r_grid])
    # Add 26th near center gap
    centers.append([0.5, 0.5])
    centers = np.array(centers[:n])
    radii = np.full(n, r_grid)
    # Make last one smaller to start valid
    radii[-1] = 0.02
    configs.append((centers, radii))
    
    # 3. Random dense pack
    centers = np.random.rand(n, 2)
    radii = np.full(n, 0.05)
    configs.append((centers, radii))
    
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Get starting points
    initial_configs = get_initial_configs(n)
    
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Define bounds for variables: x, y in [0, 1], r in [0, 0.5]
    # Each circle: x, y, r
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    for i_init, (init_centers, init_radii) in enumerate(initial_configs):
        # Flatten variables
        init_vars = np.concatenate([init_centers.flatten(), init_radii])
        
        # Try multiple optimizations per start to avoid local minima
        for k in range(3): # 3 restarts per config
            # Add small jitter
            jittered_vars = init_vars + np.random.normal(0, 1e-4, size=init_vars.shape)
            
            # Clip bounds
            for b_idx, b in enumerate(bounds):
                jittered_vars[b_idx] = np.clip(jittered_vars[b_idx], b[0], b[1])
                
            # Optimize
            res = opt.minimize(
                objective_func,
                jittered_vars,
                args=(n,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            # Extract results
            opt_centers = res.x[:2*n].reshape(-1, 2)
            opt_radii = res.x[2*n:]
            
            # Ensure radii non-negative
            opt_radii = np.maximum(opt_radii, 0)
            
            # Calculate actual sum
            current_sum = np.sum(opt_radii)
            
            # Check validity loosely (penalty should be low)
            # We will strictly validate at the end, but here we track sum
            if current_sum > best_sum:
                # Quick check if it's not completely broken
                if compute_overlap_penalty(opt_centers, opt_radii) < 1e-3:
                    best_sum = current_sum
                    best_centers = opt_centers.copy()
                    best_radii = opt_radii.copy()

    if best_centers is None:
        # Fallback to simple valid pack
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        for i in range(n):
            best_centers[i] = [0.5, 0.5]
            best_radii[i] = 0.0

    # Final Safety & Refinement
    # Shrink radii slightly if there are numerical overlaps to ensure strict validity
    # Then try to expand slightly if space allows? 
    # Simpler: Just validate and shrink if needed.
    
    # Compute pairwise distances
    dists = np.sqrt(np.sum((best_centers[:, np.newaxis] - best_centers[np.newaxis, :])**2, axis=2) + 1e-15)
    min_dists = np.min(dists[np.triu_indices(n, k=1)])
    
    # Required separation
    req_sep = best_radii[:, np.newaxis] + best_radii[np.newaxis, :]
    min_req = np.min(req_sep[np.triu_indices(n, k=1)])
    
    # If overlap, scale down radii
    if min_dists < min_req - 1e-12:
        scale = min_dists / (min_req + 1e-9)
        best_radii *= np.maximum(scale, 0.9999) # Shrink
        
    # Boundary check
    # Ensure x+r <= 1, etc.
    # If violation, shrink radii based on boundary distance
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        max_r = min(x, 1-x, y, 1-y)
        if max_r < r:
            best_radii[i] = max(0, max_r - 1e-12)

    # Sort circles by radius descending for tidiness (optional)
    # Not required but helps debugging
    
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum
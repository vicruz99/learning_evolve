# sol_000084 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state aba87625) state=b0a06f4a sum of radii=1.712564 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_gradients(centers, radii):
    """Compute gradient of the overlap/boundary penalty function w.r.t centers."""
    N = centers.shape[0]
    grad_c = np.zeros_like(centers)
    penalty = 0.0
    
    # Pairwise constraints
    diffs = centers[:, None, :] - centers[None, :, :]  # Shape (N, N, 2)
    dists = np.linalg.norm(diffs, axis=2) + 1e-12      # Avoid div by zero
    np.fill_diagonal(dists, np.inf)                    # Ignore self
    reqs = radii[:, None] + radii[None, :]             # Required distances
    
    overlaps = reqs - dists
    mask = overlaps > 1e-9
    active_overlaps = np.where(mask, overlaps, 0.0)
    
    # Gradient contribution from overlaps: -2 * overlap / dist * (c_i - c_j)
    if np.any(mask):
        inv_dists = np.where(mask, 1.0 / dists, 0.0)
        weights = -2.0 * active_overlaps * inv_dists
        grad_c += np.sum(weights[:, :, np.newaxis] * diffs, axis=1)
        
    penalty += np.sum(active_overlaps**2)
    
    # Boundary constraints (x and y)
    for d in range(2):
        # Left/Bottom boundary: x_i - r_i >= 0  -> violation when r_i - x_i > 0
        viol_low = radii - centers[:, d]
        mask_low = viol_low > 1e-9
        grad_c[:, d] += np.where(mask_low, -2.0 * viol_low, 0.0)
        penalty += np.where(mask_low, viol_low**2, 0.0).sum()
        
        # Right/Top boundary: x_i + r_i <= 1  -> violation when x_i + r_i - 1 > 0
        viol_high = centers[:, d] + radii - 1.0
        mask_high = viol_high > 1e-9
        grad_c[:, d] += np.where(mask_high, 2.0 * viol_high, 0.0)
        penalty += np.where(mask_high, viol_high**2, 0.0).sum()
        
    return grad_c, penalty

def run_packing():
    N = 26
    best_sum = 0.0
    best_centers = np.zeros((N, 2))
    best_radii = np.zeros(N)
    
    # Try a few different initial seeds/perturbations for robustness
    for seed in range(3):
        np.random.seed(42 + seed)
        
        # 1. Hexagonal Lattice Initialization
        centers = []
        r_init = 0.065
        y = r_init
        row = 0
        while len(centers) < N:
            x = r_init
            if row % 2 == 1:
                x += r_init  # Shift odd rows for hexagonal packing
            while x + r_init <= 1.0 and len(centers) < N:
                centers.append([x, y])
                x += 2 * r_init
            y += np.sqrt(3) * r_init
            row += 1
            
        centers = np.array(centers)
        # Add small random perturbation to break symmetries
        centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
        centers = np.clip(centers, 0, 1)
        radii = np.full(N, r_init)
        
        # 2. Optimization Loop
        lr_pos = 0.04
        rad_growth = 0.0003
        penalty_threshold = 1e-5
        
        for step in range(12000):
            grad_c, penalty = compute_gradients(centers, radii)
            
            # Gradient descent on positions
            centers -= lr_pos * grad_c
            centers = np.clip(centers, 0, 1)
            
            # Adapt radii based on validity
            if penalty < penalty_threshold:
                radii += rad_growth
            else:
                radii -= rad_growth * 0.2  # Retract slightly if invalid
                
            # Decay schedules
            lr_pos *= 0.9992
            rad_growth *= 0.9995
            
            # Track best
            current_sum = np.sum(radii)
            if current_sum > best_sum and penalty < 1e-4:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    # 3. Final Safety Projection
    # Shrink radii minimally to guarantee strict satisfaction of 1e-12 tolerance
    # This handles any residual numerical drift from the optimization loop
    for _ in range(200):
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < best_radii[i] + best_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
            for d_coord in range(2):
                if best_centers[i, d_coord] - best_radii[i] < -1e-12 or \
                   best_centers[i, d_coord] + best_radii[i] > 1 + 1e-12:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            break
        best_radii *= 0.9999  # Extremely conservative shrink step
        
    return best_centers, best_radii, np.sum(best_radii)

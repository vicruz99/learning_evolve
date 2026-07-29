# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b5cb09ab) state=f8b0d3e6 sum of radii=1.962264 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def compute_sum_radii(centers):
    """
    Computes the sum of maximum feasible radii for a given set of centers.
    Assumes circles are packed such that they don't overlap and are inside [0,1]^2.
    For each circle, radius is limited by distance to walls and distance to other circles.
    Note: This function calculates radii independently based on distances, 
    which is valid for the sum if we consider the radii as variables determined by the tightest constraints.
    However, strictly speaking, radii are coupled. But for the purpose of finding a valid packing 
    with high sum, we can compute r_i based on current positions as a heuristic or final step.
    
    For the optimization of positions to maximize equal radii r, we use:
    r = min( dist_to_wall, min_pairwise_dist / 2 )
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Precompute distances to walls
    # Wall distances for circle i
    walls_dist = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1 - centers[:, 1]))
    
    # Compute pairwise distances
    # centers is (n, 2)
    # diff is (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf) # Distance to self is infinity
    
    # Radius for circle i is limited by min wall distance and half min pairwise distance
    min_pair_dists = np.min(dists, axis=1)
    radii = np.minimum(walls_dist, min_pair_dists / 2.0)
    
    return np.sum(radii)

def objective_equal_radii(centers_flat):
    """
    Objective function to minimize for equal radii optimization.
    We want to maximize r, where r = min( min_pair_dist/2, min_wall_dist ).
    Equivalent to minimizing -r.
    """
    centers = centers_flat.reshape(26, 2)
    
    # Check bounds penalty (soft constraint)
    # Ideally we keep centers in [0, 1], but optimizer might drift.
    # We add a penalty if outside.
    penalty = 0.0
    if np.any(centers < 0) or np.any(centers > 1):
        penalty = 1000.0 * (np.sum(np.minimum(centers, 0)**2) + np.sum(np.maximum(centers - 1, 0)**2))
        # Clip for distance calc to avoid NaNs or weirdness if far out?
        # Actually if far out, dists are large, r is large? No, wall dist becomes negative.
        # Let's clip for calculation logic.
        centers = np.clip(centers, 1e-9, 1 - 1e-9)

    # Calculate min pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    # Calculate min wall distance
    # Distance to left/right/bottom/top
    d_left = centers[:, 0]
    d_right = 1 - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = 1 - centers[:, 1]
    wall_dists = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    min_wall_dist = np.min(wall_dists)

    # The radius r is constrained by both
    # If min_pair_dist < 2 * min_wall_dist, r = min_pair_dist / 2
    # Else r = min_wall_dist
    # We want to maximize this r.
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    return -r + penalty

def run_packing():
    # Number of circles
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We want to place n points.
    # Estimate spacing. Area ~ 1. Density ~ 0.9. 
    # 26 * pi * r^2 ~ 0.9 => r ~ 0.105. Diameter ~ 0.21.
    # Spacing ~ 0.21.
    # Let's try a hexagonal pattern.
    
    centers = []
    r_est = 0.1 # Initial radius guess
    
    # Rows of hexagonal packing
    # Vertical spacing: r * sqrt(3)
    # Horizontal spacing: 2r
    # Offset for odd rows: r
    
    # We need to fit 26 points.
    # Let's try to fill rows.
    row_idx = 0
    while len(centers) < n:
        y = r_est + row_idx * r_est * math.sqrt(3)
        if y + r_est > 1: # Check if row fits vertically (approx)
            # If row doesn't fit, maybe adjust r_est or stop?
            # But we are just placing points, optimizer will fix it.
            pass
            
        # X positions
        if row_idx % 2 == 0:
            x_start = r_est
        else:
            x_start = r_est + r_est # offset by r_est (half of 2r)
            
        x = x_start
        while x + r_est <= 1 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_est
        
        row_idx += 1
        
    # If we generated more or fewer, adjust. 
    # With r_est=0.1, we likely got enough points or need to trim.
    # Let's ensure we have exactly 26.
    centers = centers[:n]
    
    # If we have fewer than 26 (unlikely with 0.1), we can add random points or adjust.
    # But 0.1 is small, so we should have plenty.
    # Let's convert to numpy array
    centers = np.array(centers)
    
    # 2. Optimization
    # We optimize the positions to maximize the minimum separation (and wall distance).
    # Flatten centers for optimizer
    x0 = centers.flatten()
    
    # Use Powell or Nelder-Mead. Powell is often better for non-smooth.
    # Bounds: all coords in [0, 1]
    bounds = [(0, 1)] * (n * 2)
    
    # Nelder-Mead doesn't support bounds well, so we rely on initialization and maybe penalty.
    # Powell supports bounds? No, Powell in scipy doesn't support bounds directly in older versions, 
    # but 'Nelder-Mead' and 'Powell' are derivative free.
    # 'L-BFGS-B' supports bounds but needs gradient (or finite diff).
    # 'TNC' supports bounds.
    
    # Let's try 'L-BFGS-B' with finite difference gradient? 
    # Or just 'Nelder-Mead' with a tight penalty for boundaries.
    
    # Let's use 'Nelder-Mead' as it's robust for non-smooth objectives.
    # We rely on the penalty in objective function to keep points inside.
    
    result = minimize(objective_equal_radii, x0, method='Nelder-Mead', 
                     options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-9})
    
    opt_centers = result.x.reshape(n, 2)
    
    # 3. Compute Radii
    # Now that we have optimal positions, compute the exact radii.
    # We allow radii to be unequal to squeeze out more sum, though they will be close.
    # Actually, for the sum objective, if positions are optimized for equal radii,
    # computing max feasible radii for each individually might be inconsistent (overlaps might occur if we just take min dist / 2).
    # Wait. If r_i = min(dist_to_wall, min_j dist(i,j)/2), then r_i + r_j <= dist(i,j) is guaranteed?
    # r_i <= dist(i,j)/2 and r_j <= dist(i,j)/2 => r_i + r_j <= dist(i,j). Yes.
    # And r_i <= dist_to_wall. Yes.
    # So this calculation yields a valid packing.
    
    # Recompute radii
    radii = np.zeros(n)
    diff = opt_centers[:, np.newaxis, :] - opt_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dists = np.min(dists, axis=1)
    
    walls_dist = np.minimum(np.minimum(opt_centers[:, 0], 1 - opt_centers[:, 0]), 
                            np.minimum(opt_centers[:, 1], 1 - opt_centers[:, 1]))
    
    radii = np.minimum(walls_dist, min_pair_dists / 2.0)
    
    sum_radii = np.sum(radii)
    
    return opt_centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    # Check validity manually if needed
    # print(validate_packing(centers, radii))

# sol_000062 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 608ae89b) state=ca40dfe9 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses a force-directed layout algorithm starting from a hexagonal grid.
    """
    n = 26
    sqrt3 = math.sqrt(3)
    
    # 1. Initialization
    # Pattern: 4, 5, 4, 5, 4, 4 rows (Hexagonal packing)
    # This pattern fits well in a square with aspect ratio close to 1.
    points = []
    # Row 0: 4 circles
    for x in [0, 2, 4, 6]:
        points.append((x, 0))
    # Row 1: 5 circles
    for x in [1, 3, 5, 7, 9]:
        points.append((x, sqrt3))
    # Row 2: 4 circles
    for x in [0, 2, 4, 6]:
        points.append((x, 2*sqrt3))
    # Row 3: 5 circles
    for x in [1, 3, 5, 7, 9]:
        points.append((x, 3*sqrt3))
    # Row 4: 4 circles
    for x in [0, 2, 4, 6]:
        points.append((x, 4*sqrt3))
    # Row 5: 4 circles
    for x in [1, 3, 5, 7]:
        points.append((x, 5*sqrt3))
        
    pts = np.array(points)
    
    # Scale and center to fit in [0, 1] square
    # The bounding box of pts is roughly [0, 9] x [0, 8.66]
    # We scale uniformly to fit with some margin.
    # Max dimension is 9.
    # Scale to fit width 0.9 (leaving 0.05 margin on sides)
    # But we need to center it.
    
    width = 9.0
    height = 5.0 * sqrt3 # approx 8.66
    
    # Scale factor to fit within [0.05, 0.95] range?
    # Actually, let's just fit to [0, 1] first then optimize.
    # But to avoid immediate boundary issues, fit to [0.05, 0.95].
    # Available space 0.9.
    scale = 0.9 / width
    
    pts_scaled = pts * scale
    
    # Center in [0, 1]
    # Current range [0, 0.9] in x, [0, 0.78] in y
    # We want center at 0.5.
    # x offset: 0.5 - 0.9/2 = 0.05
    # y offset: 0.5 - 0.78/2 = 0.5 - 0.39 = 0.11
    
    offset = np.array([0.05, 0.11])
    centers = pts_scaled + offset
    
    # 2. Optimization (Force-directed)
    steps = 3000
    k_repulse = 100.0 
    k_boundary = 500.0
    dt = 0.0005
    
    # Preallocate forces
    forces = np.zeros((n, 2))
    
    for step in range(steps):
        # Cooling schedule
        temp = 1.0 - (step / steps)
        current_dt = dt * (0.1 + 0.9 * temp)
        
        # Compute pairwise repulsive forces
        # Vectorized
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dists_sq_safe = np.maximum(dists_sq, 1e-12)
        dists = np.sqrt(dists_sq_safe)
        
        # Force magnitude: k / dist^2 (Coulomb-like)
        # But to prevent explosion, use softening or clip
        # F = k * vector / dist^3
        # vector / dist^3 = diff / (dists * dists_sq)
        
        # Compute inv_dist_cubed
        inv_dist_cubed = 1.0 / (dists * dists_sq_safe + 1e-15)
        
        # Forces contribution from all pairs
        # F_ij = k * diff / dist^3
        # Sum over j
        # diff shape (N, N, 2)
        # inv_dist_cubed shape (N, N)
        # Result shape (N, 2)
        
        # We need to sum forces exerted by j on i.
        # Force on i from j is proportional to (c_i - c_j).
        # So we sum over j (axis 1).
        
        pair_forces = diff * inv_dist_cubed[:, :, np.newaxis] * k_repulse
        forces = np.sum(pair_forces, axis=1)
        
        # Boundary forces
        # Push back if outside [0, 1]
        # Force proportional to distance from boundary
        
        # X boundary
        # If x < 0, force + (0-x) = -x
        # If x > 1, force + (1-x)
        x = centers[:, 0]
        y = centers[:, 1]
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Left wall
        mask_left = x < 0.0
        fx[mask_left] = k_boundary * (-x[mask_left])
        
        # Right wall
        mask_right = x > 1.0
        fx[mask_right] = k_boundary * (1.0 - x[mask_right])
        
        # Bottom wall
        mask_bottom = y < 0.0
        fy[mask_bottom] = k_boundary * (-y[mask_bottom])
        
        # Top wall
        mask_top = y > 1.0
        fy[mask_top] = k_boundary * (1.0 - y[mask_top])
        
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        # Update centers
        centers += forces * current_dt
        
        # Hard clip to keep strictly inside [0, 1] for validity check later?
        # Actually, boundary forces should keep them in.
        # But clipping is safer.
        centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
        
    # 3. Calculate Radii
    # Radius of circle i is limited by distance to boundaries and other circles.
    radii = np.full(n, 1.0)
    
    # Boundary constraints
    radii[:] = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    radii[:] = np.minimum(radii, centers[:, 1])
    radii[:] = np.minimum(radii, 1.0 - centers[:, 1])
    
    # Circle-circle constraints
    # r_i <= dist(i, j) / 2 for all j != i
    # This is equivalent to r_i + r_j <= dist(i, j) IF r_i = r_j = dist/2?
    # No.
    # We need to find r_i such that r_i + r_j <= dist_ij.
    # If we set r_i = min_j (dist_ij / 2), then for any j:
    # r_i <= dist_ij / 2
    # r_j <= dist_ij / 2 (since r_j is min over k of dist_ik/2, including i)
    # So r_i + r_j <= dist_ij.
    # Thus, setting r_i = min_j (dist_ij / 2) is a valid assignment.
    # Is it optimal?
    # For a fixed set of centers, the max radius for circle i is indeed min(dist_ij/2, boundaries).
    # Because if r_i > dist_ij/2, then r_i + r_j > dist_ij (assuming r_j >= 0).
    # Wait, if r_j is very small, maybe r_i can be larger?
    # Constraint: r_i + r_j <= dist.
    # If r_j = 0, r_i <= dist.
    # But r_j is determined by its own constraints.
    # If we set all r_i to their max possible values independently (min over j of dist_ij/2),
    # we might violate r_i + r_j <= dist?
    # Let's check.
    # Suppose 3 circles in line. A, B, C.
    # d(A,B) = 2, d(B,C) = 2, d(A,C) = 4.
    # Max r_A based on B: 1. Based on C: 2. Min = 1.
    # Max r_B based on A: 1. Based on C: 1. Min = 1.
    # Max r_C based on B: 1. Based on A: 2. Min = 1.
    # Result r_A=1, r_B=1, r_C=1.
    # Check A-C: 1+1 <= 4. OK.
    # Generally, triangle inequality holds?
    # dist(A,C) <= dist(A,B) + dist(B,C).
    # If r_A <= d_AB/2 and r_C <= d_BC/2? No.
    # r_C <= d_AC/2.
    # We need r_A + r_C <= d_AC.
    # We have r_A <= d_AB/2, r_C <= d_AC/2.
    # Is d_AB/2 + d_AC/2 <= d_AC?
    # d_AB/2 <= d_AC/2 <=> d_AB <= d_AC.
    # This is true if B is closer to A than C is? Not necessarily.
    # But we take min over ALL j.
    # So r_A <= d_AC/2.
    # And r_C <= d_AC/2.
    # So r_A + r_C <= d_AC.
    # Yes.
    # Because r_A is bounded by d_AC/2 directly (since C is a neighbor).
    # So the independent assignment r_i = min_j (dist_ij / 2) is always valid.
    # And it maximizes each r_i individually?
    # Yes, because r_i cannot exceed dist_ij/2 for any j.
    # So this is the maximal radius vector for fixed centers.
    
    # Compute distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    
    # Min distance to any other circle
    min_dists = np.min(dists, axis=1)
    
    # Update radii
    radii[:] = np.minimum(radii, min_dists / 2.0)
    
    # Sum of radii
    sum_radii = np.sum(radii)
    
    # Validate manually to be sure (optional but good for debugging)
    # But we trust the logic.
    
    return centers, radii, sum_radii

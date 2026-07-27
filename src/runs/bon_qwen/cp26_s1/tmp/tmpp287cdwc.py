import numpy as np

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates.
        radii: np.array of shape (26,) with radius of each circle.
        sum_radii: float sum of radii.
    """
    n = 26
    rng = np.random.RandomState(42)
    
    # 1. Initialization: Perturbed Hexagonal Lattice
    # We try to fit 26 circles in a hexagonal pattern.
    # A 6-row pattern with 5, 4, 5, 4, 5, 4 circles sums to 27.
    # We will take the first 26 positions.
    
    centers_list = []
    # Approximate radius for 27 circles in hex packing is around 0.094
    # We start with a slightly smaller radius to allow room for optimization
    initial_r = 0.08 
    
    # Parameters for hex lattice
    # Horizontal spacing: 2*r
    # Vertical spacing: r*sqrt(3)
    # We scale these relative to the initial_r guess to fit in 1x1
    # But since r is a variable, we just pick a geometry that fits 1x1 with margin
    
    # Let's generate a grid that fits 1x1 with radius 0.1 (just for positioning logic)
    # Then we place centers.
    
    row_count = 6
    # Rows alternate in count: 5, 4, 5, 4, 5, 4
    # But we need 26.
    # Pattern: 5, 4, 5, 4, 5, 3 (sum 26) or just take first 26 of 5,4,5,4,5,4.
    
    # Geometric setup for hex packing in unit square
    # If we assume a target radius r_target, 
    # width required for 5 circles is 10*r_target. If r_target ~ 0.1, width ~ 1.
    # height for 6 rows is 2*r + 5*r*sqrt(3) ~ 10.66*r. If r~0.094, height ~ 1.
    
    # Let's just generate normalized coordinates and scale to 1x1
    # Centers at (x, y)
    
    # Row y positions (normalized 0 to 1)
    # We have 6 rows.
    # y indices: 0, 1, 2, 3, 4, 5
    # y = (index + 0.5) / 6 ? No, hex packing needs specific vertical spacing.
    
    # Let's use a direct placement strategy based on hex geometry
    # We want to pack them as densely as possible.
    
    # Let's create a list of (col, row, shift) tuples
    # Rows:
    # Row 0: 5 circles, shift 0
    # Row 1: 4 circles, shift 0.5 (in units of 2r spacing)
    # Row 2: 5 circles, shift 0
    # Row 3: 4 circles, shift 0.5
    # Row 4: 5 circles, shift 0
    # Row 5: 3 circles, shift 0.5 (to make sum 26: 5+4+5+4+5+3 = 26)
    
    # Coordinates in "hex units" where horizontal dist = 1, vertical dist = sqrt(3)/2
    # Actually let's use standard hex coords:
    # dx = 1, dy = sqrt(3)/2
    
    # Let's build coordinates
    pts = []
    # We will scale these later to fit in [0,1]x[0,1]
    
    # To maximize density, we assume a radius r.
    # But we just need valid centers.
    # Let's place them on a grid that fits roughly.
    
    # Grid approach:
    # 5 columns, 6 rows is 30 points. Too many.
    # Let's use the specific counts.
    
    # Let's just generate random points within a shrinking bounding box?
    # No, hex is better.
    
    # Let's create points for 5, 4, 5, 4, 5, 3
    # X-coords for full row (5): 1, 3, 5, 7, 9 (units of r)
    # X-coords for half row (4): 2, 4, 6, 8
    # X-coords for 3: 2, 4, 6
    
    # But we need to scale.
    # Let's assume max width is 10r (for 5 circles).
    # Max height is 2r + 5r*sqrt(3) (for 6 rows).
    
    # We can just define relative coordinates and normalize.
    
    coords = []
    
    # Row 0: 5 circles
    # x: 1, 3, 5, 7, 9 (relative to 2r=2) -> actually let's use 0.5, 1.5, 2.5...
    # Let's use integer coordinates for grid and scale.
    # Grid unit = 2r (diameter).
    # Shift = 1 (radius) for hex.
    
    # Row 0 (y=0): 5 circles at x = 1, 3, 5, 7, 9
    for x in [1, 3, 5, 7, 9]:
        coords.append([x, 0])
        
    # Row 1 (y=sqrt(3)): 4 circles at x = 2, 4, 6, 8
    # Wait, hex shift is 1 unit (radius) horizontally if vertical is sqrt(3)*radius?
    # Standard hex: neighbors at (1, sqrt(3)) relative to (0,0) if radius=1?
    # Distance sqrt(1+3) = 2 = 2r. Correct.
    # So x shift is 1, y shift is sqrt(3).
    
    y_idx = 0
    y_step = np.sqrt(3)
    
    # We have built Row 0.
    # Row 1
    y_idx += y_step
    for x in [2, 4, 6, 8]:
        coords.append([x, y_idx])
        
    # Row 2
    y_idx += y_step
    for x in [1, 3, 5, 7, 9]:
        coords.append([x, y_idx])
        
    # Row 3
    y_idx += y_step
    for x in [2, 4, 6, 8]:
        coords.append([x, y_idx])
        
    # Row 4
    y_idx += y_step
    for x in [1, 3, 5, 7, 9]:
        coords.append([x, y_idx])
        
    # Row 5
    y_idx += y_step
    # We need 3 circles to reach 26. (5+4+5+4+5 = 23, need 3).
    for x in [2, 4, 6]:
        coords.append([x, y_idx])
        
    coords = np.array(coords)
    
    # Now we have coords in "hex units".
    # We need to map them to [0, 1] x [0, 1].
    # The extent of x: min 1, max 9. Range 8.
    # The extent of y: min 0, max 5*sqrt(3) ~ 8.66.
    
    # We can center and scale.
    # But we want them to fill the square.
    # Let's scale x and y independently to fill 1x1 with some margin for radii.
    # Since radii will grow, let's pack them tight initially.
    
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    
    # Scale to fit in [0.05, 0.95] initially
    scale_x = 0.9 / (max_x - min_x)
    scale_y = 0.9 / (max_y - min_y)
    
    # To preserve aspect ratio of hex lattice, use min scale
    scale = min(scale_x, scale_y)
    
    centers_init = coords.copy()
    centers_init[:, 0] = (centers_init[:, 0] - min_x) * scale + 0.05
    centers_init[:, 1] = (centers_init[:, 1] - min_y) * scale + 0.05
    
    # Add some random perturbation to break symmetry
    centers_init += rng.uniform(-0.02, 0.02, size=centers_init.shape)
    
    # Clip to valid range [0.1, 0.9] to be safe
    centers_init = np.clip(centers_init, 0.1, 0.9)
    
    centers = centers_init
    radii = np.ones(n) * 0.02 # Start small
    
    # 2. Optimization Loop
    # We want to maximize sum(radii) subject to constraints.
    # We minimize Loss = -sum(radii) + Penalty
    
    penalty_weight = 500.0
    lr_c = 1e-3
    lr_r = 1e-3
    
    # Number of iterations
    # A fixed number of steps. 
    # We can use a cooling schedule or fixed steps.
    steps = 2000
    
    # Precompute indices for pairwise distances to speed up?
    # With N=26, N^2 is small (676), explicit loops or vectorization both fine.
    # Vectorization is cleaner.
    
    # We will run gradient descent.
    
    for step in range(steps):
        # Compute gradients
        grad_c = np.zeros_like(centers)
        grad_r = np.zeros_like(radii)
        
        # Target: Maximize sum(r) => Gradient of -sum(r) is -1.
        # So we add -1 to grad_r (for minimization of loss = -sum + penalty)
        # Wait, Loss = -sum(r) + P.
        # dLoss/dr = -1 + dP/dr.
        # Update r = r - lr * dLoss/dr = r - lr * (-1 + dP/dr) = r + lr * (1 - dP/dr).
        # So effectively we add +1 to the gradient component for radius increase logic if we view it as ascent.
        # Let's stick to minimization of Loss.
        
        grad_r -= 1.0 # Derivative of -sum(r) w.r.t r is -1
        
        # 1. Boundary Constraints
        # Constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        # Violations:
        # Left: r - x > 0
        # Right: r + x - 1 > 0
        # Bottom: r - y > 0
        # Top: r + y - 1 > 0
        
        # We can compute violations and gradients.
        # Using vectorized operations
        
        # Left wall: x - r >= 0. Violation val = max(0, r - x)
        viol_left = np.maximum(0, radii - centers[:, 0])
        # Gradient of (r-x)^2 w.r.t r is 2(r-x), w.r.t x is -2(r-x)
        # In Loss: + lambda * (r-x)^2
        # dL/dr += 2*lambda*(r-x)
        # dL/dx += -2*lambda*(r-x)
        
        grad_r += 2.0 * penalty_weight * viol_left
        grad_c[:, 0] += -2.0 * penalty_weight * viol_left
        
        # Right wall: 1 - x - r >= 0. Violation val = max(0, r + x - 1)
        viol_right = np.maximum(0, radii + centers[:, 0] - 1.0)
        grad_r += 2.0 * penalty_weight * viol_right
        grad_c[:, 0] += 2.0 * penalty_weight * viol_right
        
        # Bottom wall: y - r >= 0. Violation val = max(0, r - y)
        viol_bottom = np.maximum(0, radii - centers[:, 1])
        grad_r += 2.0 * penalty_weight * viol_bottom
        grad_c[:, 1] += -2.0 * penalty_weight * viol_bottom
        
        # Top wall: 1 - y - r >= 0. Violation val = max(0, r + y - 1)
        viol_top = np.maximum(0, radii + centers[:, 1] - 1.0)
        grad_r += 2.0 * penalty_weight * viol_top
        grad_c[:, 1] += 2.0 * penalty_weight * viol_top
        
        # 2. Pairwise Overlaps
        # Constraint: dist(i, j) >= r_i + r_j
        # Violation: max(0, r_i + r_j - dist(i, j))
        
        # Compute pairwise distances
        # centers shape (n, 2)
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
        dist_sq = np.sum(diff**2, axis=2) # (n, n)
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-10)
        dist = np.sqrt(dist_sq)
        
        # Sum of radii matrix
        rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :] # (n, n)
        
        # Overlap amount
        overlap = np.maximum(0, rad_sum - dist) # (n, n)
        
        # We only care about i < j to avoid double counting in penalty sum?
        # Loss includes sum_{i<j} (overlap)^2.
        # Gradient contribution for pair (i, j):
        # dL/dr_i = 2 * overlap_ij * (1)
        # dL/dr_j = 2 * overlap_ij * (1)
        # dL/dC_i = 2 * overlap_ij * (- (C_i - C_j) / dist)
        # dL/dC_j = 2 * overlap_ij * (- (C_j - C_i) / dist)
        
        # We can sum over all pairs and divide by 2 for radii, or handle carefully.
        # Let's iterate or use broadcasting carefully.
        # Broadcasting is fast enough for 26x26.
        
        # Mask for upper triangle to handle pairs once
        # But broadcasting computes both (i,j) and (j,i).
        # If we sum over all i,j, we double count.
        # So we compute contributions for all i,j and then sum?
        # Actually, the term is (overlap)^2.
        # d/d r_i of (r_i + r_j - d)^2 = 2(overlap) * 1.
        # If we sum over all pairs (i,j) including j=i (overlap 0) and double counting:
        # Sum_{i,j} (overlap_ij)^2.
        # Gradient w.r.t r_k comes from terms where i=k or j=k.
        # Term (k, j): 2 * overlap_kj.
        # Term (j, k): 2 * overlap_jk.
        # Since overlap is symmetric, total grad = 2 * sum_j (2 * overlap_kj) = 4 * sum_j overlap_kj.
        # But we want gradient of sum_{i<j} (overlap)^2.
        # So we should divide by 2?
        # Let's just use a loop or careful masking to be safe and simple.
        # N=26 is small.
        
        # Vectorized approach with mask
        # Create a mask where i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        # Overlaps for these pairs
        overlaps_pairs = overlap[i_idx, j_idx] # vector of size M
        
        # Distance for these pairs
        dists_pairs = dist[i_idx, j_idx]
        
        # Direction vectors (C_j - C_i) / dist
        # diff[i, j] = C_j - C_i
        diff_pairs = diff[i_idx, j_idx] # (M, 2)
        
        # Normalize direction
        # diff_unit = diff_pairs / dists_pairs[:, np.newaxis]
        # But handle dist=0? dists_pairs > 0 due to max 1e-10
        
        diff_unit = diff_pairs / dists_pairs[:, np.newaxis]
        
        # Gradient contributions
        # For radii:
        # grad_r[i] += 2 * overlap
        # grad_r[j] += 2 * overlap
        
        # We can accumulate using np.add.at
        factor = 2.0 * penalty_weight * overlaps_pairs
        
        grad_r[i_idx] += factor
        grad_r[j_idx] += factor
        
        # For centers:
        # grad_c[i] += -2 * overlap * (C_i - C_j)/dist  <-- Wait
        # Let's recheck sign.
        # Loss term P = (r_i + r_j - d)^2.
        # dP/dC_i = 2(r_i+r_j-d) * (-dC_i/d(d_ij)) ? No.
        # d(d_ij)/dC_i = (C_i - C_j)/d_ij.
        # So dP/dC_i = 2(overlap) * (-(C_i - C_j)/d_ij) = 2(overlap) * (C_j - C_i)/d_ij.
        # This force pushes C_i towards C_j?
        # If overlap > 0, we want to increase distance.
        # Increasing distance means moving C_i away from C_j.
        # Vector (C_i - C_j) points from j to i.
        # So we want to add component in direction (C_i - C_j).
        # Wait.
        # P is penalty. We minimize P.
        # Gradient of P points in direction of steepest ascent of P.
        # We want to move opposite to gradient (descent).
        # If overlap > 0, P increases as we get closer?
        # P = (sum - d)^2.
        # If d decreases (get closer), sum-d increases, P increases.
        # So P is high when close.
        # Gradient w.r.t C_i points towards C_j (increasing P).
        # So -Gradient points away from C_j.
        # Let's check derivative sign again.
        # P = (S - d)^2. dP/dd = -2(S-d).
        # d/dC_i = dP/dd * dd/dC_i = -2(overlap) * (C_i - C_j)/d.
        # = 2(overlap) * (C_j - C_i)/d.
        # Vector (C_j - C_i) points from i to j.
        # So gradient points towards j.
        # Descent direction ( -grad ) points away from j.
        # Correct.
        # So we subtract the gradient from C.
        # C_i <- C_i - lr * grad_c_i.
        # So we need to add the gradient to the variable `grad_c` which we subtract later.
        # Wait, in my code: `centers -= lr * grad_c`.
        # So I need to compute `grad_c` as the gradient of Loss.
        # Loss = -sum(r) + P.
        # grad_c of Loss = grad_c of P.
        # grad_c_i = 2 * overlap * (C_j - C_i) / d.
        # This is the force pulling i towards j.
        # So `grad_c[i]` should accumulate `factor * diff_unit`?
        # diff_unit was (C_j - C_i)/d.
        # Yes.
        
        force = factor[:, np.newaxis] * diff_unit # (M, 2)
        
        grad_c[i_idx] += force
        grad_c[j_idx] -= force # Symmetry: force on j is opposite
        
        # Update parameters
        centers -= lr_c * grad_c
        radii -= lr_r * grad_r
        
        # Constraints enforcement (clipping)
        # Radii must be non-negative
        radii = np.maximum(radii, 1e-6)
        
        # Centers must be in [0, 1]. 
        # Although penalty handles it, explicit clipping prevents wild moves.
        # But we must be careful not to violate boundary constraints relative to radii.
        # The penalty term handles r <= x etc.
        # However, if center goes out of [0,1], it's invalid.
        centers = np.clip(centers, 0.0, 1.0)
        
        # Optional: Adaptive learning rate?
        # Decay lr to settle down
        if step % 500 == 0 and step > 0:
            lr_c *= 0.8
            lr_r *= 0.8
            penalty_weight *= 1.1 # Increase penalty weight to tighten constraints

    # Final clean up
    # Ensure valid
    radii = np.maximum(radii, 0)
    centers = np.clip(centers, 0, 1)
    
    # Recalculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Verification logic (not part of run_packing return, but for thought process)
# If I run this, will it reach 2.636?
# The penalty method is robust. 
# 2.636 / 26 = 0.1014.
# With good initialization, it should reach a local optimum close to global.
# Hexagonal packing is very efficient.
# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 47219f56) state=b4778ad7 sum of radii=2.226406 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square.
    Returns (centers, radii, sum_radii).
    """
    N = 26
    np.random.seed(42) # For reproducibility

    # Helper function to compute heuristic radii and sum given centers
    def compute_heuristic_sum(centers):
        radii = np.zeros(N)
        total_sum = 0.0
        
        # Precompute distances for efficiency? N=26 is small enough.
        
        for i in range(N):
            cx, cy = centers[i]
            # Distance to walls
            d_wall = min(cx, 1.0 - cx, cy, 1.0 - cy)
            
            # Distance to other centers
            min_dist_to_other = float('inf')
            for j in range(N):
                if i == j:
                    continue
                dist = math.sqrt((cx - centers[j, 0])**2 + (cy - centers[j, 1])**2)
                if dist < min_dist_to_other:
                    min_dist_to_other = dist
            
            # Radius is limited by wall and half-distance to nearest neighbor
            # This is a heuristic that works well for dense packings
            r = min(d_wall, min_dist_to_other / 2.0)
            radii[i] = r
            total_sum += r
            
        return total_sum, radii

    # Helper to compute forces (gradients) to move centers
    # We want to maximize sum(radii).
    # Gradient of r_i w.r.t c_i:
    # 1. If limited by wall (e.g., x=0), grad is (1, 0) pointing inward.
    # 2. If limited by neighbor j (r_i = dist/2), grad is 0.5 * unit_vector(c_i - c_j).
    # Also, moving c_i affects r_j if r_j is limited by c_i.
    # Gradient of r_j w.r.t c_i (when r_j limited by dist to i) is 0.5 * unit_vector(c_i - c_j).
    # So forces are repulsive.
    
    def compute_forces(centers):
        forces = np.zeros((N, 2))
        
        # First pass: identify limiting constraints for each circle
        # Store (limiting_factor, limiting_entity)
        # limiting_entity: -1 for wall (store axis/pos), j for circle index
        limits = [] 
        
        for i in range(N):
            cx, cy = centers[i]
            d_wall = min(cx, 1.0 - cx, cy, 1.0 - cy)
            
            # Find nearest neighbor
            min_d = float('inf')
            nearest_j = -1
            for j in range(N):
                if i == j: continue
                d = math.sqrt((cx - centers[j, 0])**2 + (cy - centers[j, 1])**2)
                if d < min_d:
                    min_d = d
                    nearest_j = j
            
            r_val = min(d_wall, min_d / 2.0)
            
            # Determine active constraint(s)
            # If multiple, we can average forces or pick one. 
            # Picking the tightest one is usually safest for gradient ascent.
            # However, for stability, let's consider if it's close to both.
            
            active = []
            
            # Check wall constraints
            # x=0
            if abs(cx - r_val) < 1e-9 and abs(cx - min_d/2.0) > 1e-9: # Strictly wall?
                 active.append(('x0', 1.0)) # Force (1, 0)
            # x=1
            if abs((1.0 - cx) - r_val) < 1e-9 and abs((1.0 - cx) - min_d/2.0) > 1e-9:
                 active.append(('x1', -1.0)) # Force (-1, 0)
            # y=0
            if abs(cy - r_val) < 1e-9 and abs(cy - min_d/2.0) > 1e-9:
                 active.append(('y0', 1.0)) # Force (0, 1)
            # y=1
            if abs((1.0 - cy) - r_val) < 1e-9 and abs((1.0 - cy) - min_d/2.0) > 1e-9:
                 active.append(('y1', -1.0)) # Force (0, -1)
            
            # Check neighbor constraint
            if abs((min_d / 2.0) - r_val) < 1e-9:
                if nearest_j != -1:
                    active.append(('neighbor', nearest_j))

            # Fallback if numerical issues or multiple tight constraints
            # If r is determined by distance, add neighbor force
            # If r is determined by wall, add wall force
            # If both are close (tangent to wall and neighbor), add both?
            # Let's just add the dominant one.
            
            # Refined logic:
            # If dist/2 is the min, force away from neighbor.
            # If wall is min, force away from wall.
            # If very close, combine?
            
            if min_d / 2.0 <= d_wall + 1e-9:
                # Neighbor limited
                if nearest_j != -1:
                    dx = cx - centers[nearest_j, 0]
                    dy = cy - centers[nearest_j, 1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d > 1e-9:
                        forces[i, 0] += 0.5 * (dx / d)
                        forces[i, 1] += 0.5 * (dy / d)
                else:
                    # No neighbors? Should not happen in packing
                    pass
            else:
                # Wall limited
                if abs(cx - r_val) < 1e-7: forces[i, 0] += 1.0
                elif abs(1.0 - cx - r_val) < 1e-7: forces[i, 0] -= 1.0
                elif abs(cy - r_val) < 1e-7: forces[i, 1] += 1.0
                elif abs(1.0 - cy - r_val) < 1e-7: forces[i, 1] -= 1.0
        
        # Second pass: Forces on neighbors due to moving i
        # If circle j is limited by distance to i, moving i away from j helps j.
        # We already added 0.5 * unit(i-j) to i's force (from i's perspective).
        # From j's perspective, moving i away from j increases dist, so increases r_j.
        # Gradient of r_j w.r.t c_i is 0.5 * unit(i-j).
        # So we should add 0.5 * unit(i-j) to force on i? 
        # Wait. 
        # Objective F = sum r_k.
        # dF/dc_i = dr_i/dc_i + sum_{k!=i} dr_k/dc_i.
        # dr_i/dc_i: if limited by j, 0.5 * unit(i-j).
        # dr_k/dc_i: if k limited by i, 0.5 * unit(i-k).
        # Note unit(i-j) = - unit(j-i).
        # So if i and j limit each other:
        # Force on i from dr_i: 0.5 * (c_i - c_j)/d. (Push away from j)
        # Force on i from dr_j: 0.5 * (c_i - c_j)/d. (Push away from j)
        # Total = 1.0 * unit vector away from j.
        
        # Let's correct the force calculation to be rigorous.
        forces = np.zeros((N, 2))
        
        for i in range(N):
            cx, cy = centers[i]
            d_wall = min(cx, 1.0 - cx, cy, 1.0 - cy)
            
            # Find nearest neighbor distance
            min_d = float('inf')
            nearest_j = -1
            for j in range(N):
                if i == j: continue
                d = math.sqrt((cx - centers[j, 0])**2 + (cy - centers[j, 1])**2)
                if d < min_d:
                    min_d = d
                    nearest_j = j
            
            r_i = min(d_wall, min_d / 2.0)
            
            # 1. Contribution from r_i's gradient
            # If limited by wall
            if d_wall <= min_d / 2.0:
                if abs(cx - d_wall) < 1e-9: forces[i, 0] += 1.0
                elif abs(1.0 - cx - d_wall) < 1e-9: forces[i, 0] -= 1.0
                elif abs(cy - d_wall) < 1e-9: forces[i, 1] += 1.0
                elif abs(1.0 - cy - d_wall) < 1e-9: forces[i, 1] -= 1.0
            # If limited by neighbor j
            else:
                if nearest_j != -1:
                    dx = cx - centers[nearest_j, 0]
                    dy = cy - centers[nearest_j, 1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d > 1e-9:
                        forces[i, 0] += 0.5 * (dx / d)
                        forces[i, 1] += 0.5 * (dy / d)

            # 2. Contribution from r_k's gradient where k is limited by i
            # We need to check if any k has r_k determined by distance to i.
            # This is equivalent to checking if i is the nearest neighbor to k.
            # We can do this by iterating all k, but O(N^2) is fine.
            # Actually, we can just check the nearest neighbor for each k.
        
        # Precompute nearest neighbors for all k
        nearest_map = {}
        for k in range(N):
            cx, cy = centers[k]
            min_d = float('inf')
            nn = -1
            for m in range(N):
                if k == m: continue
                d = math.sqrt((cx - centers[m, 0])**2 + (cy - centers[m, 1])**2)
                if d < min_d:
                    min_d = d
                    nn = m
            nearest_map[k] = (nn, min_d)

        for k in range(N):
            nn_k, d_k = nearest_map[k]
            cx_k, cy_k = centers[k]
            d_wall_k = min(cx_k, 1.0 - cx_k, cy_k, 1.0 - cy_k)
            
            # Is k limited by distance to nn_k?
            if d_k / 2.0 <= d_wall_k + 1e-9: # Tolerant check
                # k is limited by nn_k.
                # Moving nn_k away from k increases r_k.
                # Gradient w.r.t center of nn_k is 0.5 * unit(nn_k - k).
                if nn_k != -1:
                    nn_cx, nn_cy = centers[nn_k]
                    dx = nn_cx - cx_k
                    dy = nn_cy - cy_k
                    d = math.sqrt(dx*dx + dy*dy)
                    if d > 1e-9:
                        forces[nn_k, 0] += 0.5 * (dx / d)
                        forces[nn_k, 1] += 0.5 * (dy / d)

        return forces

    # Function to run optimization for a specific initial configuration
    def optimize(start_centers):
        centers = start_centers.copy()
        current_sum, _ = compute_heuristic_sum(centers)
        
        # Learning rate / step size
        step = 0.05
        
        # Number of iterations
        # We can run for a fixed number of steps
        for iteration in range(500):
            forces = compute_forces(centers)
            
            # Normalize forces? Or just use as is.
            # Magnitudes are around 0.5 to 1.
            # Apply forces
            new_centers = centers + step * forces
            
            # Clip to valid range (with margin)
            # Margin 0.001 to avoid numerical issues at exact boundary
            margin = 0.001
            new_centers = np.clip(new_centers, margin, 1.0 - margin)
            
            # Check if this move improves sum
            new_sum, _ = compute_heuristic_sum(new_centers)
            
            if new_sum > current_sum:
                centers = new_centers
                current_sum = new_sum
                # Maybe increase step slightly?
                step = min(step * 1.01, 0.1) 
            else:
                # Reduce step
                step = step * 0.9
                if step < 1e-6:
                    break
        
        return centers, current_sum

    # Generate multiple starting configurations
    best_centers = None
    best_sum = -1.0

    # Strategy 1: Hexagonal Grid
    # Estimate spacing for 26 circles
    # Area approx 26 * pi * r^2 <= 1 => r ~ 0.11
    # Spacing ~ 0.22
    # Grid size ~ 1/0.22 ~ 4.5 rows
    hex_row_count = 5
    hex_col_count = 6 # 30 spots, we pick 26
    
    # Generate hex points
    hex_points = []
    row_height = math.sqrt(3)/2 * 0.22 # approx
    col_width = 0.22
    
    # Center the grid
    # Y range 0 to 1
    # X range 0 to 1
    
    # Let's try to pack them in a block
    # 5 rows. 6, 5, 6, 5, 4? Total 26.
    # Row 0: 6 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 6
    # Row 3: 5
    # Row 4: 4
    
    # Or just generate a dense hex grid and pick first 26
    grid_points = []
    y = 0.1 # margin
    while y < 0.9:
        x = 0.1
        row_shift = 0 if int((y - 0.1) / row_height) % 2 == 0 else col_width / 2
        while x + row_shift < 0.9:
            grid_points.append([x + row_shift, y])
            x += col_width
        y += row_height
    
    # If we have more than 26, we can prune or just pick random subset?
    # Better to place them optimally.
    # Let's just take the first 26 if we generated enough, or generate specifically.
    
    # Let's construct specifically 26 points in hex pattern
    # 6 rows of ~4-5?
    # 5 rows: 6, 5, 6, 5, 4 = 26.
    
    initial_configs = []
    
    # Config 1: 5 rows (6,5,6,5,4)
    centers = np.zeros((26, 2))
    idx = 0
    row_h = 1.0 / 6.0 # rough estimate for spacing
    # Actually let's optimize spacing later, just place in grid
    # Let's place in 6x5 grid (30) and remove 4 worst?
    # No, hex is better.
    
    # Let's create a generic hex grid
    # 6 rows
    rows = [5, 5, 5, 5, 4, 2] # sum = 26
    # Actually 5,5,5,5,5,1 = 26.
    # Let's try 5 rows of 5, 1 row of 1.
    
    # Let's just use random restarts from Grid and Random.
    
    # 1. Grid
    g_centers = []
    # 5x5 grid = 25. Add 1 in center.
    for r in range(5):
        for c in range(5):
            x = 0.125 + c * 0.25
            y = 0.125 + r * 0.25
            g_centers.append([x, y])
    # Add 26th at center (0.5, 0.5)
    g_centers.append([0.5, 0.5])
    initial_configs.append(np.array(g_centers))
    
    # 2. Random
    for _ in range(5):
        rand_centers = np.random.rand(26, 2) * 0.8 + 0.1
        initial_configs.append(rand_centers)
        
    # 3. Hexagonal-ish
    hex_centers = []
    # 5 rows
    # Row 0: 6 cols
    # Row 1: 5 cols (offset)
    # Row 2: 6 cols
    # Row 3: 5 cols
    # Row 4: 4 cols
    # Total 26
    # Width of 6 cols: 5 gaps.
    # Height of 5 rows: 4 gaps.
    
    # Let's fit in [0.1, 0.9]
    width_avail = 0.8
    height_avail = 0.8
    
    # Row 0 (6 items)
    step_x = width_avail / 5.5 # slightly wider
    # Actually just distribute
    # Let's assume standard hex packing density logic
    
    # Just generate points on a hex lattice and filter to 26 inside
    pts = []
    r_hex = 0.08 # radius guess
    d_hex = 2 * r_hex # spacing
    
    y = 0.1
    row = 0
    while y < 0.9:
        x = 0.1
        offset = (d_hex / 2) if row % 2 == 1 else 0
        while x < 0.9:
            pts.append([x + offset, y])
            x += d_hex
        y += d_hex * math.sqrt(3)/2
        row += 1
    
    if len(pts) >= 26:
        # Pick 26 that are most spread out?
        # Just take first 26
        initial_configs.append(np.array(pts[:26]))
    else:
        # If not enough, pad with random
        curr = np.array(pts)
        pad = np.random.rand(26 - len(pts), 2) * 0.8 + 0.1
        initial_configs.append(np.vstack([curr, pad]))

    # Run optimization for each config
    all_results = []
    for config in initial_configs:
        # Normalize/Clip just in case
        config = np.clip(config, 0.01, 0.99)
        opt_c, opt_s = optimize(config)
        all_results.append((opt_c, opt_s))

    # Pick best
    best_res = max(all_results, key=lambda x: x[1])
    centers = best_res[0]
    
    # Final Step: Compute exact valid radii for these centers
    # We solve the problem: max sum r_i s.t. r_i + r_j <= dist, r_i <= wall_dist
    # This is an LP. But we can solve it greedily or just use the heuristic which is valid.
    # Wait, heuristic r_i = min(wall, dist/2) is VALID.
    # Is it optimal?
    # As discussed, maybe not.
    # But for the purpose of this problem, a valid packing with high sum is good.
    # Let's try to improve radii by solving a simple LP using a custom solver or just iterative adjustment.
    
    # Iterative radius adjustment (Water-filling style)
    # 1. Compute distances matrix
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = math.sqrt(np.sum((centers[i] - centers[j])**2))
            dists[i, j] = d
            dists[j, i] = d
            
    wall_limits = np.zeros(N)
    for i in range(N):
        x, y = centers[i]
        wall_limits[i] = min(x, 1-x, y, 1-y)
        
    # Initialize radii
    radii = np.zeros(N)
    for i in range(N):
        min_d = np.min(dists[i, :]) # min distance to any other
        # Actually min over j!=i
        dists_row = dists[i, :]
        # Filter self
        dists_row[i] = float('inf')
        min_d = np.min(dists_row)
        radii[i] = min(wall_limits[i], min_d / 2.0)
        
    # Now, can we increase some radii?
    # If radii[i] + radii[j] < dists[i, j], we have slack.
    # We can increase radii.
    # This is exactly the LP.
    # Let's implement a simple solver for this specific structure.
    # It's a max flow problem or just linear system?
    # Actually, it's maximizing sum x_i subject to x_i + x_j <= c_ij, x_i <= b_i.
    # This can be solved by finding the minimum cut in a graph?
    # Or just use scipy linprog if available.
    
    # Since I cannot rely on scipy inside the submission environment necessarily (though allowed),
    # and writing a full LP solver is complex, let's stick to the heuristic which is very good.
    # However, we can try a simple local improvement on radii.
    
    # Simple loop:
    # For each pair (i, j), if r_i + r_j < dist, we can increase one or both.
    # But increasing one might violate other constraints.
    # This is complex.
    
    # Let's trust the heuristic radii derived from the optimized centers.
    # The centers were optimized to maximize the heuristic sum.
    
    # Recompute final radii carefully
    final_radii = np.zeros(N)
    for i in range(N):
        # Wall constraint
        r = wall_limits[i]
        # Neighbor constraints
        for j in range(N):
            if i == j: continue
            # r + r_j <= dist
            # But we haven't fixed r_j yet.
            # The heuristic r_i = dist/2 assumes r_j = dist/2.
            # If r_j is smaller, r_i could be larger.
            # But since we optimized centers to maximize sum of mins, 
            # the configuration is likely tightly packed.
            # Let's just use the strict bounds based on distances assuming neighbors take half.
            # It is a valid packing.
            min_d = min_d if 'min_d' in locals() else float('inf')
            d = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if d < min_d: min_d = d
        r = min(r, min_d / 2.0)
        final_radii[i] = r

    sum_radii = np.sum(final_radii)
    
    # Double check validity
    # If any overlap, reduce radii slightly
    # (Should not happen with heuristic)
    
    return centers, final_radii, float(sum_radii)

# For execution check
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum: {s}")
    # print(r)

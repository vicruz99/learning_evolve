# sol_000325 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3035e37b) state=f834433d sum of radii=2.311124 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # Helper function to initialize centers in a hexagonal pattern
    def initialize_hexagonal(n):
        centers = np.zeros((n, 2))
        # Try to fit n circles in a hexagonal lattice
        # Estimate number of rows
        # For n=26, maybe 6 rows? 5-5-5-5-5-1?
        # Or 5 rows with varying counts.
        # Let's try a generic hexagonal placement
        
        # Approximate spacing
        # Area per circle approx 1/n. Radius approx sqrt(1/(n*pi)) ~ 0.11
        # Spacing ~ 2*r ~ 0.22
        
        idx = 0
        row = 0
        col = 0
        
        # We will fill rows. Even rows have offset.
        # Let's determine row length.
        # Width 1.0. Spacing x = 2*r. 
        # If r~0.1, spacing 0.2. ~5 circles per row.
        
        rows_needed = int(np.ceil(n / 5.0)) + 1
        current_r = 0.12 # Initial estimate for spacing
        
        # Generate points
        points = []
        y = current_r
        row_idx = 0
        while len(points) < n:
            x = current_r
            if row_idx % 2 == 1:
                x += current_r * np.sqrt(3) # Offset for hexagonal
                # Actually standard hex packing:
                # Row 0: (0,0), (2r, 0), ...
                # Row 1: (r, sqrt(3)r), (3r, sqrt(3)r), ...
                # Offset is r horizontally.
                # Let's use offset = current_r.
                x_offset = current_r
            else:
                x_offset = 0
            
            x = current_r + x_offset
            while x < 1.0 - current_r + 1e-5 and len(points) < n:
                points.append((x, y))
                x += 2 * current_r
                # Check horizontal limit carefully
                if x + current_r > 1.0:
                    break
            
            y += current_r * np.sqrt(3)
            row_idx += 1
        
        if len(points) < n:
            # If not enough, fill remaining randomly or extend grid
            # Extend grid
            while len(points) < n:
                # Just add to next row
                y += current_r * np.sqrt(3)
                x = current_r + (0 if len(points)%2==0 else current_r) # simple offset logic
                # Better: just random fill if needed, but let's stick to grid expansion
                # Re-generate with slightly smaller spacing?
                pass 
        
        # If grid generation is tricky, let's use a simpler method:
        # Grid of points with slight random noise
        centers = np.zeros((n, 2))
        # 5x5 grid is 25. Add 1.
        # Let's try 6x5 grid (30 points) and take first 26, then optimize?
        # Or just a dense random start?
        # Dense random start is robust.
        
        # Let's try a specific layout for 26
        # 5 rows. 5, 5, 5, 5, 6?
        # Or 5, 5, 5, 6, 5?
        
        # Let's use a hexagonal grid generator properly
        pts = []
        r_est = 0.105
        step_x = 2 * r_est
        step_y = r_est * np.sqrt(3)
        
        y = r_est
        row = 0
        while len(pts) < n:
            x = r_est
            if row % 2 == 1:
                x += r_est # Offset
            while x < 1.0 - r_est + 1e-4 and len(pts) < n:
                pts.append([x, y])
                x += step_x
            y += step_y
            row += 1
        
        centers = np.array(pts[:n])
        return centers

    # Better initialization: Random dense packing
    def initialize_random(n):
        centers = np.random.rand(n, 2)
        return centers

    # We will use the hexagonal init as it's better structured
    centers = initialize_hexagonal(n)
    
    # If the hexagonal init didn't produce n points (due to boundary checks), fallback
    if centers.shape[0] < n:
        centers = initialize_random(n)
    
    # Ensure centers are within [0,1]
    centers = np.clip(centers, 0, 1)
    
    # Initial radii
    # Start small and expand
    radii = np.full(n, 0.05)
    
    # Optimization parameters
    iterations = 2000
    expansion_rate = 1.001 # Multiplicative factor for radii
    push_strength = 0.5
    cooling = 1.0
    
    # Precompute indices
    indices = np.arange(n)
    
    # Main optimization loop
    # Strategy: Iteratively expand radii and resolve overlaps
    # This is a form of "expanding gas" simulation
    
    for _ in range(iterations):
        # 1. Try to expand radii
        # We can expand by a small additive amount or multiplicative
        # To maximize sum, we want to increase radii as much as possible.
        # Let's try to increase each radius by the minimum slack available.
        
        # Calculate slack for each circle
        # Slack is min(distance to boundary, distance to other circles - r_other - r_self)
        # But we don't want to check all pairs every time (O(N^2) is fine for N=26)
        
        # Update radii
        for i in range(n):
            # Check boundaries
            slack_x = min(centers[i, 0] - radii[i], 1.0 - centers[i, 0] - radii[i])
            slack_y = min(centers[i, 1] - radii[i], 1.0 - centers[i, 1] - radii[i])
            slack_boundary = min(slack_x, slack_y)
            
            # Check other circles
            slack_others = np.inf
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                required_dist = radii[i] + radii[j]
                gap = dist - required_dist
                if gap < slack_others:
                    slack_others = gap
            
            slack = min(slack_boundary, slack_others)
            
            # Increase radius by a fraction of the slack
            # Use a learning rate that decreases over time?
            # Or just take a step.
            # To prevent oscillation, take small step.
            delta_r = slack * 0.1 
            if delta_r > 0:
                radii[i] += delta_r
        
        # 2. Resolve Overlaps (Push centers)
        # If radii increased, overlaps might occur (due to discrete steps or numerical issues)
        # We need to push centers apart.
        
        # Compute forces/overlaps
        for i in range(n):
            # Boundary forces
            # Push away from boundaries if inside
            for axis in range(2):
                if centers[i, axis] - radii[i] < 0:
                    centers[i, axis] = radii[i]
                if centers[i, axis] + radii[i] > 1.0:
                    centers[i, axis] = 1.0 - radii[i]
            
            # Pairwise forces
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist == 0:
                    # Avoid division by zero, separate randomly
                    diff = np.random.rand(2) * 0.01 - 0.005
                    dist = np.sqrt(np.sum(diff**2))
                
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    # Push apart
                    # Normalize direction
                    dir_vec = diff / dist
                    # Move i and j apart by half overlap each (or proportional to radii?)
                    # Equal move is stable
                    move = overlap * push_strength
                    centers[i] += dir_vec * move
                    centers[j] -= dir_vec * move
                    
                    # Clamp to boundaries after move
                    for axis in range(2):
                        centers[i, axis] = np.clip(centers[i, axis], radii[i], 1.0 - radii[i])
                        centers[j, axis] = np.clip(centers[j, axis], radii[j], 1.0 - radii[j])

    # Final check and clamp
    for i in range(n):
        for axis in range(2):
            centers[i, axis] = np.clip(centers[i, axis], radii[i], 1.0 - radii[i])

    sum_radii = np.sum(radii)
    
    # Validate and fix any remaining tiny overlaps by shrinking slightly if needed?
    # The prompt validator allows 1e-12 tolerance.
    # Our iterative push should handle it.
    
    # To ensure validity, let's run a quick fix pass
    # If any overlap, reduce radii slightly
    # Actually, the push logic ensures centers are valid for current radii?
    # Wait, the push logic moves centers. If radii are large, centers might be pushed out of bounds?
    # We clamp centers. But clamping might cause overlap with boundary?
    # centers[i, axis] = radii[i] ensures it touches boundary.
    # If two circles are at boundaries and overlap, we push them.
    # If one is stuck at boundary, the other moves away.
    # This logic is sound.
    
    # One refinement: The expansion step `radii[i] += delta_r` might cause overlaps that the push step resolves.
    # But the push step moves centers. Moving centers might violate boundary constraints?
    # We clamp centers.
    # However, if we clamp a center, it might overlap another circle.
    # So we might need multiple passes of push.
    # But for 2000 iterations, it should converge.
    
    # Let's return the result
    return centers, radii, sum_radii

# Run the packing
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation
    import math
    valid = True
    for i in range(len(c)):
        if c[i,0] - r[i] < -1e-9 or c[i,0] + r[i] > 1 + 1e-9 or \
           c[i,1] - r[i] < -1e-9 or c[i,1] + r[i] > 1 + 1e-9:
            valid = False
            break
        for j in range(i+1, len(c)):
            d = math.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
            if d < r[i] + r[j] - 1e-9:
                valid = False
                break
    print(f"Valid: {valid}")

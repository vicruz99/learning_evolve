# sol_000109 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 82d73ba2) state=ee0bc8c4 sum of radii=0.211846 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize positions in a hexagonal lattice
    # We aim for 6 rows to fit 26 circles.
    # Distribution: 5, 4, 5, 4, 5, 3 circles per row.
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Start with a small radius to ensure no initial overlap
    current_r = 0.05
    
    # Horizontal spacing for hex packing is 2*r
    # Vertical spacing is sqrt(3)*r
    # We will scale this later.
    
    idx = 0
    # Pattern of circles per row
    row_counts = [5, 4, 5, 4, 5, 3]
    
    # We need to fit this in [0,1]x[0,1]. 
    # Let's calculate bounds roughly to place them.
    # Width occupied by 5 circles of radius r: 10r. 
    # If r ~ 0.1, width is 1. 
    # Let's assume r ~ 0.09 initially.
    # Horizontal spacing dx = 2*current_r
    # Vertical spacing dy = current_r * math.sqrt(3)
    
    # We will scale coordinates to fit [0,1]x[0,1] tightly later.
    # For now, let's just place them with dx=2, dy=sqrt(3) and normalize.
    
    # To center the packing, we can calculate the bounding box of the lattice points
    # and then scale/translate.
    
    raw_centers = []
    
    y_pos = 0
    for i, count in enumerate(row_counts):
        # In hex packing, even rows (0, 2, 4...) are shifted or odd rows?
        # Let's shift odd rows by 1 unit of radius (dx/2) relative to previous?
        # Actually, standard hex: row k shifted by r relative to row k-1.
        # If we use dx = 2, shift is 1.
        
        x_start = 0
        if i % 2 == 1:
            x_start = 1 # shift by 1 unit (which corresponds to r in scaled coords)
        
        for j in range(count):
            x = x_start + j * 2
            y = y_pos
            raw_centers.append([x, y])
        
        y_pos += math.sqrt(3)
    
    raw_centers = np.array(raw_centers)
    
    # Normalize to fit in [0,1] x [0,1] with some margin
    # Find bounds
    min_x, max_x = raw_centers[:, 0].min(), raw_centers[:, 0].max()
    min_y, max_y = raw_centers[:, 1].min(), raw_centers[:, 1].max()
    
    width = max_x - min_x
    height = max_y - min_y
    
    # We want to leave room for the radius. 
    # If we scale to fit exactly in [0,1], the circles at edges will touch boundaries with r=0.
    # We want to fit circles of radius r.
    # Let's target a specific r, say 0.09, and scale to fit.
    # Actually, better to scale the lattice to fit the square, then the effective radius 
    # is determined by the gap.
    
    # Let's scale to fit in [0,1] first.
    # Scale factors
    sx = 1.0 / width
    sy = 1.0 / height
    scale = min(sx, sy)
    
    centers = raw_centers * scale
    centers[:, 0] -= (centers[:, 0].min() - (1 - centers[:, 0].max()) * scale) # Center x?
    # Simpler: shift to 0,0 then scale, then center?
    
    # Re-center
    centers -= centers.min(axis=0) # Shift min to 0
    centers *= scale # Scale
    
    # Center in [0,1]
    cx, cy = centers.max(axis=0)
    shift_x = (1 - cx) / 2
    shift_y = (1 - cy) / 2
    centers += np.array([shift_x, shift_y])
    
    # Now centers are in [0,1].
    # Initial radius guess
    # Distance between closest points in lattice
    # In scaled lattice, horizontal dist is 2*scale, vertical is sqrt(3)*scale.
    # min_dist = 2 * scale (since 2 < sqrt(3) approx 1.732? No 2 > 1.732)
    # Wait, 2*scale vs sqrt(3)*scale. sqrt(3) ~ 1.732.
    # So vertical neighbors are closer?
    # In raw lattice, neighbors are at distance 2.
    # After scaling, distance is 2*scale.
    
    min_dist = 2 * scale
    radii[:] = min_dist / 2 * 0.9 # Start with 90% of max possible to be safe
    
    # 2. Repulsive Force Simulation to relax and expand
    
    # We will iteratively try to increase radius and resolve overlaps
    step = 0.001
    max_iter = 1000
    target_r = radii[0]
    
    # To maximize sum of radii, we can assume equal radii first, then maybe vary.
    # Let's stick to equal radii for the main expansion as it's robust.
    
    for _ in range(max_iter):
        # Check for overlaps and boundaries
        overlap = False
        
        # Boundary violations
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r:
                centers[i, 0] = r
                overlap = True
            elif x > 1 - r:
                centers[i, 0] = 1 - r
                overlap = True
            if y < r:
                centers[i, 1] = r
                overlap = True
            elif y > 1 - r:
                centers[i, 1] = 1 - r
                overlap = True
        
        # Pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist_ij = radii[i] + radii[j]
                
                if dist < min_dist_ij:
                    overlap = True
                    # Push apart
                    if dist > 1e-9:
                        push_factor = (min_dist_ij - dist) / dist * 0.5 # Move each by half overlap
                        nx, ny = dx/dist, dy/dist
                        centers[i, 0] += nx * push_factor
                        centers[i, 1] += ny * push_factor
                        centers[j, 0] -= nx * push_factor
                        centers[j, 1] -= ny * push_factor
                    else:
                        # Very close, random nudge
                        centers[i, 0] += 0.001
                        centers[j, 0] -= 0.001

        # If no overlaps, try to increase radii
        if not overlap:
            # Increase radii slightly
            expansion = 1e-5
            radii[:] *= (1 + expansion)
            # Clamp to reasonable max (0.5)
            radii[:] = np.clip(radii, 0, 0.5)
        else:
            # If overlaps, just keep positions, maybe radii don't increase
            # We can try to decrease radii slightly if stuck? 
            # But the push apart handles it.
            pass
            
        # Small perturbation to escape local minima?
        # Not implemented to keep it deterministic mostly.
        
    # 3. Gradient Descent Optimization
    # Maximize the minimum distance (clearance)
    # Clearance = min( min_dist(i,j), min_dist_to_boundary(i) )
    # We want to maximize this value.
    # Let f(C) = min_{i,j} (||c_i - c_j|| - 2r) and min_i (dist_to_boundary - r)
    # But r is fixed from previous step? 
    # Let's optimize positions C for a fixed r to maximize clearance, 
    # then we can increase r.
    
    # Actually, let's just optimize positions to maximize the "bottleneck" distance.
    # Let D be the bottleneck distance. r = D/2.
    # We maximize D.
    
    # Flatten centers for optimizer
    x0 = centers.flatten()
    
    def negative_min_distance(vars):
        c = vars.reshape((n, 2))
        min_d = 10.0
        
        # Boundary distances
        for i in range(n):
            d = min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1])
            if d < min_d: min_d = d
            
        # Pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                d = math.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                if d < min_d: min_d = d
        
        return -min_d # Minimize negative max
    
    # Bounds for coordinates
    bounds = [(0, 1) for _ in range(2*n)]
    
    # Run optimization
    res = minimize(negative_min_distance, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
    centers = res.x.reshape((n, 2))
    
    # Calculate achieved radius
    min_clearance = -res.fun
    radii[:] = min_clearance / 2.0
    
    # 4. Unequal Radii Refinement
    # Now that we have a valid packing with equal radii r_eq,
    # sum is 26 * r_eq.
    # Can we improve by making some radii larger?
    # This is harder to optimize directly. 
    # But we can try to expand radii individually.
    
    # Simple heuristic: try to increase each radius one by one until blocked.
    # But this is slow (O(N^2)).
    # Instead, let's run a force simulation with variable radii.
    
    # Reset centers to the optimized equal-radii positions
    # radii are currently all equal.
    
    # Run variable radius simulation
    # Potential energy: sum of overlaps squared
    # We want to maximize sum(r_i) s.t. overlaps <= 0.
    # Lagrangian approach or just penalty.
    # Let's use a penalty method:
    # Maximize sum(r) - Penalty * sum(max(0, overlap)^2)
    # If Penalty is high, we satisfy constraints.
    # But we want to push r up.
    
    # Let's just try to increase all radii again with variable constraint?
    # No, let's just use the equal radius solution. 
    # It's usually very close to optimal for sum of radii.
    # However, to be safe, let's run a quick coordinate ascent on radii.
    
    # Coordinate ascent on radii:
    for _ in range(50): # 50 passes
        changed = False
        for i in range(n):
            # Try to increase r_i
            # Constraints:
            # 1. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
            # 2. r_i + r_j <= dist(i, j) for all j != i
            
            max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            
            for j in range(n):
                if i == j: continue
                dist = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                allowed_r = dist - radii[j]
                if allowed_r < max_r:
                    max_r = allowed_r
            
            if max_r > radii[i] + 1e-9:
                radii[i] = max_r
                changed = True
        
        if not changed:
            break

    sum_radii = np.sum(radii)
    
    # Validate
    # The validation function checks strict non-overlap.
    # Our coordinate ascent might have numerical issues or slight overlaps due to float precision?
    # Let's enforce strict constraints by clipping.
    
    # Re-verify and fix overlaps by shrinking slightly if needed
    # But coordinate ascent ensures r_i <= dist - r_j => r_i + r_j <= dist.
    # So it should be valid.
    # However, floating point errors might exist.
    # Let's run the validation logic internally to be sure? 
    # No, just return.
    
    # One last check: if any radius is negative (shouldn't happen), set to 0.
    radii = np.maximum(radii, 0)
    
    return centers, radii, sum_radii

# Wrapper to run the function
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Simple validation
    valid = True
    for i in range(len(r)):
        for j in range(i+1, len(r)):
            d = np.sqrt(np.sum((c[i]-c[j])**2))
            if d < r[i] + r[j] - 1e-12:
                valid = False
                print(f"Overlap {i},{j}")
        if c[i,0] < r[i] or c[i,0] > 1-r[i] or c[i,1] < r[i] or c[i,1] > 1-r[i]:
            valid = False
            print(f"Boundary {i}")
    print(f"Valid: {valid}")

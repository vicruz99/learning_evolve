# sol_000266 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8d1f387b) state=596b7c68 sum of radii=2.236854 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Returns a valid packing of 26 circles in a unit square maximizing the sum of radii.
    """
    n = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Helper function to check validity and return score
    def evaluate(centers, radii):
        # Check boundary
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return -1.0 # Invalid
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-9:
                    return -1.0 # Invalid
        
        return np.sum(radii)

    # Initial configuration generator
    def generate_initial_config(seed=0):
        rng = np.random.RandomState(seed)
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # Strategy: Hexagonal-like packing to fit 26 circles
        # Try to fit in 6 rows roughly
        # Row counts: 5, 4, 5, 4, 5, 3 (Sum = 26)
        # Or 5, 5, 5, 5, 4, 2? 
        # Let's try a dense grid perturbation
        
        # Base grid 5x5 = 25. We need 1 more.
        # Let's place 26 circles in a 6-row hexagonal pattern.
        # Row heights: r, r + sqrt(3)r, ...
        
        # Estimate radius
        # If 6 rows, height ~ 2r + 5*sqrt(3)r <= 1 => r(2 + 8.66) <= 1 => r <= 0.105
        # If 5 circles per row, width ~ 10r <= 1 => r <= 0.1
        # So r ~ 0.095 might be safe.
        
        r_est = 0.095
        radius = r_est
        
        count = 0
        row_idx = 0
        y_pos = radius
        
        while count < n and row_idx < 8: # Safety limit
            # Determine number of circles in this row
            # Alternating 5 and 4 to save space, but we need 26.
            # 5+4+5+4+5+4 = 27. We need 26.
            # Let's do 5, 5, 5, 5, 4, 2? No, height constraint.
            # Let's stick to 5, 4, 5, 4, 5, 3 (Total 26)
            
            if row_idx % 2 == 0:
                num_in_row = 5
                x_start = radius
            else:
                num_in_row = 4 # Shifted row
                x_start = radius + radius # Shifted by r? No, shifted by r horizontally means center at 2r?
                # In hex packing, shift is r.
                # If even row starts at r, odd row starts at 2r?
                # Distance between (r, y) and (2r, y+dy) is sqrt(r^2 + dy^2).
                # If dy = sqrt(3)r, dist = sqrt(r^2 + 3r^2) = 2r. OK.
                x_start = 2 * radius

            # Adjust x_start to fit 5 or 4 circles in [0, 1]
            # For 5 circles, span is 8r + 2r = 10r. If 10r <= 1, fits.
            # For 4 circles, span is 6r + 2r = 8r. Fits easily.
            
            # However, for 5 circles in shifted row (start 2r), 
            # x positions: 2r, 4r, 6r, 8r, 10r.
            # Max x + r = 11r. If 11r > 1, doesn't fit.
            # So shifted rows can only hold 4 circles if r=0.1.
            
            if row_idx == 5:
                num_in_row = 3 # To make sum 26? 5+4+5+4+5 = 23. Need 3 more.
            
            # Let's recalculate row counts for 26
            # 5, 4, 5, 4, 5, 3 = 26.
            # 5, 4, 5, 4, 4, 4 = 26.
            # Let's try 5, 5, 5, 5, 4, 2 (Total 26) -> 6 rows.
            # But 5 circles in row requires width 10r.
            # If r=0.09, 10r = 0.9 < 1. Fits.
            
            if row_idx < 4:
                num_in_row = 5
            elif row_idx == 4:
                num_in_row = 4
            else:
                num_in_row = 2 # Just to fill remaining
            
            # Recalculate x_start for even/odd
            # Even rows (0, 2, 4): 5 circles. x = r, 3r, 5r, 7r, 9r.
            # Odd rows (1, 3, 5): 4 circles. x = 2r, 4r, 6r, 8r.
            
            if row_idx % 2 == 0:
                num_in_row = 5
                x_positions = [radius, 3*radius, 5*radius, 7*radius, 9*radius]
            else:
                num_in_row = 4
                x_positions = [2*radius, 4*radius, 6*radius, 8*radius]
            
            # Check if we need more
            if count + num_in_row > n:
                num_in_row = n - count
                if row_idx % 2 == 0:
                    x_positions = x_positions[:num_in_row]
                else:
                    x_positions = x_positions[:num_in_row]

            for i in range(num_in_row):
                if count < n:
                    centers[count] = [x_positions[i], y_pos]
                    radii[count] = radius
                    count += 1
            
            row_idx += 1
            y_pos += math.sqrt(3) * radius # Vertical pitch for hex packing

        # Random perturbation
        centers += rng.uniform(-0.01, 0.01, size=(n, 2))
        # Clip to valid range
        centers = np.clip(centers, radius, 1 - radius)
        
        return centers, radii

    # Optimization function
    def optimize(centers, radii, steps=2000, learning_rate=1e-4):
        current_r = radii.copy()
        current_c = centers.copy()
        
        # We want to maximize sum(r). 
        # Equivalent to minimizing -sum(r).
        # We can use a penalty method.
        # Objective: -sum(r_i) + Penalty(overlaps) + Penalty(boundary)
        
        # Simple iterative inflation and relaxation
        # 1. Try to increase radii
        # 2. Resolve conflicts by moving centers
        
        # Initial radii inflation
        scale = 1.0
        min_dist = float('inf')
        
        # Check initial validity
        # If invalid, shrink radii
        for i in range(100):
            valid = True
            min_dist = float('inf')
            for j in range(n):
                x, y = current_c[j]
                r = current_r[j]
                if x < r or x > 1-r or y < r or y > 1-r:
                    valid = False
            
            if not valid:
                current_r *= 0.9
                continue
                
            for j in range(n):
                for k in range(j+1, n):
                    d = np.sqrt(np.sum((current_c[j] - current_c[k])**2))
                    req = current_r[j] + current_r[k]
                    if d < req:
                        valid = False
                        if d < min_dist: min_dist = d
            
            if valid:
                break
            else:
                current_r *= 0.95 # Shrink if overlap

        # Now expand
        for step in range(steps):
            # Calculate forces
            forces = np.zeros((n, 2))
            radius_growth = np.ones(n)
            
            # Boundary forces
            for i in range(n):
                x, y = current_c[i]
                r = current_r[i]
                if x - r < 1e-5:
                    forces[i, 0] += (x - r) * 1000 # Push right
                elif x + r > 1 - 1e-5:
                    forces[i, 0] -= (1 - (x + r)) * 1000 # Push left
                
                if y - r < 1e-5:
                    forces[i, 1] += (y - r) * 1000
                elif y + r > 1 - 1e-5:
                    forces[i, 1] -= (1 - (y + r)) * 1000

            # Overlap forces
            overlap_penalty = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = current_c[j, 0] - current_c[i, 0]
                    dy = current_c[j, 1] - current_c[i, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-9
                    req = current_r[i] + current_r[j]
                    
                    if dist < req:
                        # Overlap detected
                        overlap = req - dist
                        # Force proportional to overlap
                        fx = (dx / dist) * overlap * 100
                        fy = (dy / dist) * overlap * 100
                        forces[i] -= np.array([fx, fy])
                        forces[j] += np.array([fx, fy])
                        overlap_penalty += overlap
            
            # Apply forces
            current_c += forces * learning_rate
            
            # Clip positions to stay roughly inside (soft clip)
            # Actually, forces should keep them in. 
            # But let's enforce hard constraint for safety in next step
            # current_c = np.clip(current_c, 0, 1) # Dangerous if r > 0
            
            # Try to increase radii if overlap is small
            if overlap_penalty < 1e-4:
                growth_factor = 1.0 + 1e-5 * (1 / (1 + step/100))
                current_r *= growth_factor
            else:
                # If high overlap, maybe shrink slightly to recover?
                # Or just rely on forces to separate.
                pass
            
            # Keep radii positive
            current_r = np.maximum(current_r, 1e-6)
            
            # Ensure centers are valid for current radii (simple projection)
            for i in range(n):
                current_c[i, 0] = np.clip(current_c[i, 0], current_r[i], 1 - current_r[i])
                current_c[i, 1] = np.clip(current_c[i, 1], current_r[i], 1 - current_r[i])

        return current_c, current_r, np.sum(current_r)

    # Try multiple seeds
    final_centers = None
    final_radii = None
    max_sum = -1.0

    for seed in range(5): # 5 attempts
        c, r = generate_initial_config(seed)
        # Initial radii might be too large or small, optimize will handle
        opt_c, opt_r, score = optimize(c, r, steps=3000, learning_rate=5e-5)
        
        # Validate strictly
        valid = True
        for i in range(n):
            x, y = opt_c[i]
            rad = opt_r[i]
            if x < rad - 1e-9 or x > 1 - rad + 1e-9 or y < rad - 1e-9 or y > 1 - rad + 1e-9:
                valid = False
                break
            for j in range(i+1, n):
                d = np.sqrt(np.sum((opt_c[i]-opt_c[j])**2))
                if d < opt_r[i] + opt_r[j] - 1e-9:
                    valid = False
                    break
            if not valid: break
        
        if valid and score > max_sum:
            max_sum = score
            final_centers = opt_c
            final_radii = opt_r

    return final_centers, final_radii, max_sum

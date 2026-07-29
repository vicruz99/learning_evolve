# sol_000010 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b5cb09ab) state=8c4518c0 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Initial configuration: Hexagonal packing approximation
    # We try to fit 26 circles. A 5-row layout might work well.
    # Distribution: 6, 5, 6, 5, 4 = 26 circles
    # This layout utilizes the hexagonal offset to pack tighter.
    
    # Initial radius guess slightly above 0.1
    r_init = 0.101
    
    # Height calculation for hexagonal packing
    # Vertical distance between rows is sqrt(3) * r
    # Width available is 1.
    
    # Let's try a 5-row structure: 6, 5, 6, 5, 4
    # Row 0: 6 circles
    # Row 1: 5 circles (offset)
    # Row 2: 6 circles
    # Row 3: 5 circles (offset)
    # Row 4: 4 circles
    
    rows = [6, 5, 6, 5, 4]
    row_idx = 0
    col_idx = 0
    
    # Adjust r_init to ensure initial fit inside square for optimization start
    # Approximate height needed: 2*r + 4 * (sqrt(3)*r)
    # 2*r + 6.928*r = 8.928*r <= 1 => r <= 0.112
    # Width for 6 circles: 2*r + 5*(2*r) is not right for hex.
    # In hex, horizontal spacing is r.
    # 6 circles in a row with offset: width approx 2*r + 5*r = 7r? 
    # Actually, for 6 circles in a zig-zag or hex row, the span is larger.
    # Let's just use a grid start and let optimization fix it.
    
    # Simple grid initialization to ensure valid start
    r_start = 0.08
    centers_list = []
    
    # Fill grid-like to ensure no initial overlap
    # 6x5 grid = 30 spots, we take first 26
    # Spacing
    x_step = 1.0 / 7.0
    y_step = 1.0 / 6.0
    
    count = 0
    for r in range(6):
        for c in range(5):
            if count < n:
                x = c * x_step + x_step/2
                y = r * y_step + y_step/2
                centers_list.append([x, y])
                count += 1
    
    centers = np.array(centers_list)
    radii = np.full(n, r_start)

    # Optimization loop
    # We will iteratively increase radii and adjust positions
    
    lr = 0.001 # Learning rate for positions
    lr_r = 0.0001 # Learning rate for radius expansion
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    for step in range(5000):
        # Calculate gradients for repulsion (overlaps) and boundary forces
        # We want to maximize sum of radii.
        # Equivalent to minimizing negative sum of radii.
        # But we have constraints.
        
        # Simple repulsion dynamics
        force = np.zeros_like(centers)
        overlap_penalty = 0
        
        # 1. Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                force[i, 0] += (r - x) * 1000
            elif x - r < 1e-4:
                force[i, 0] += (1e-4 - (x - r)) * 100
                
            # Right wall
            if x + r > 1:
                force[i, 0] -= ((x + r) - 1) * 1000
            elif 1 - (x + r) < 1e-4:
                force[i, 0] -= (1e-4 - (1 - (x + r))) * 100

            # Bottom wall
            if y - r < 0:
                force[i, 1] += (r - y) * 1000
            elif y - r < 1e-4:
                force[i, 1] += (1e-4 - (y - r)) * 100

            # Top wall
            if y + r > 1:
                force[i, 1] -= ((y + r) - 1) * 1000
            elif 1 - (y + r) < 1e-4:
                force[i, 1] -= (1e-4 - (1 - (y + r))) * 100

        # 2. Circle-Circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Repulsive force proportional to overlap
                    fx = (dx / dist) * overlap * 1000
                    fy = (dy / dist) * overlap * 1000
                    
                    force[i, 0] -= fx
                    force[i, 1] -= fy
                    force[j, 0] += fx
                    force[j, 1] += fy
                    overlap_penalty += overlap
                elif dist < 1e-9:
                    # Same position, push apart randomly
                    angle = np.random.uniform(0, 2 * math.pi)
                    force[i, 0] += math.cos(angle) * 10
                    force[i, 1] += math.sin(angle) * 10
                    force[j, 0] -= math.cos(angle) * 10
                    force[j, 1] -= math.sin(angle) * 10

        # Update positions
        centers += force * lr
        
        # Clamp centers to valid range (loosely, forces handle it)
        # But ensure we don't go way out
        centers[:, 0] = np.clip(centers[:, 0], 0, 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, 1)

        # Try to expand radii
        # Check max possible expansion
        # We can increase radii if not touching boundaries or each other too tightly
        # A simple heuristic: increase all radii by a small amount, then resolve overlaps
        
        # Check if we can increase radius
        # Find minimum clearance
        min_clearance = 1.0
        
        # Boundary clearance
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            clear = min(x - r, 1 - (x + r), y - r, 1 - (y + r))
            if clear < min_clearance:
                min_clearance = clear
        
        # Inter-circle clearance
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                clear = dist - sum_r
                if clear < min_clearance:
                    min_clearance = clear
        
        # If there is space, expand radii
        if min_clearance > 1e-6:
            expansion = min(min_clearance * 0.5, 0.001) # Conservative expansion
            radii += expansion
        else:
            # If tight, maybe shrink slightly to allow rearrangement?
            # Or just rely on positional optimization to find better config
            pass
            
        # Decay learning rate
        if step % 500 == 0 and step > 0:
            lr *= 0.5
            lr_r *= 0.5

    # Final cleanup to ensure strict validity
    # If any overlaps remain due to numerical noise, shrink radii slightly
    # But we want to maximize sum, so only shrink if necessary
    
    # Re-check and adjust
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            valid = False
            break
    
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
    
    # If not valid, fallback to a known valid packing (e.g. grid)
    if not valid:
        # Construct a safe grid packing
        # 6 rows, 5 cols = 30. Take 26.
        # r = 1 / (2 * 6) = 1/12 approx 0.0833 is safe for 6 rows?
        # Actually 1/(2*rows) is for diameter.
        # If 6 rows, 2r*6 <= 1 -> r <= 1/12.
        # If 5 cols, 2r*5 <= 1 -> r <= 1/10.
        # Limiting factor is rows. r = 0.08.
        
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.08)
        
        count = 0
        r = 0.08
        for row in range(6):
            for col in range(5):
                if count < n:
                    x = col * 2 * r + r
                    y = row * 2 * r + r
                    centers[count] = [x, y]
                    count += 1
        
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Validate
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any() or np.isnan(radii).any():
            return False
        for i in range(n):
            if radii[i] < 0:
                return False
            x, y = centers[i]
            r_val = radii[i]
            if x - r_val < -1e-12 or x + r_val > 1 + 1e-12 or y - r_val < -1e-12 or y + r_val > 1 + 1e-12:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12:
                    return False
        return True

    print("Valid:", validate_packing(c, r))

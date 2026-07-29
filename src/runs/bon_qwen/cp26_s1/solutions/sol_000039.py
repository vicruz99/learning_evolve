# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2740085f) state=f025b168 sum of radii=2.540000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing():
    n = 26
    # Initial hexagonal packing setup
    # Pattern: 5, 6, 5, 6, 4
    row_counts = [5, 6, 5, 6, 4]
    centers = np.zeros((n, 2))
    
    # Start with a safe radius
    r = 0.095
    
    # Calculate initial positions
    # Vertical spacing: r * sqrt(3)
    # Horizontal spacing: 2 * r
    y = r
    total_h = 2*r + (len(row_counts)-1)*r*np.sqrt(3)
    y_offset = (1 - total_h) / 2
    y += y_offset
    
    idx = 0
    for i, count in enumerate(row_counts):
        w = 2*r*count
        x_offset = (1 - w) / 2
        x = x_offset + r
        
        # Stagger rows: shift odd rows by r
        if i % 2 == 1:
            x += r
            
        for j in range(count):
            centers[idx] = [x + j * 2 * r, y]
            idx += 1
        y += r * np.sqrt(3)
        
    # Optimization: Iteratively try to increase r and resolve conflicts
    for _ in range(200):
        r += 0.0005 # Attempt to grow
        
        # Resolve overlaps for current r
        for _ in range(100):
            forces = np.zeros((n, 2))
            valid = True
            
            for i in range(n):
                # Boundary constraints
                if centers[i, 0] < r: forces[i, 0] += (r - centers[i, 0]) * 10
                if centers[i, 0] > 1 - r: forces[i, 0] -= (centers[i, 0] - (1 - r)) * 10
                if centers[i, 1] < r: forces[i, 1] += (r - centers[i, 1]) * 10
                if centers[i, 1] > 1 - r: forces[i, 1] -= (centers[i, 1] - (1 - r)) * 10
                
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    
                    req_dist = 2 * r
                    if dist < req_dist - 1e-12:
                        valid = False
                        if dist > 1e-9:
                            overlap = req_dist - dist
                            fx = (dx / dist) * overlap * 5.0
                            fy = (dy / dist) * overlap * 5.0
                            forces[i] += [fx, fy]
                            forces[j] -= [fx, fy]
            
            centers += forces * 0.01
            centers = np.clip(centers, r, 1 - r)
            
            if valid:
                break
        
        if not valid:
            r -= 0.0001 # Shrink if we can't resolve
            
    radii = np.full(n, r)
    
    # Final validation and fallback
    if not validate_packing(centers, radii):
        # Fallback to a known valid packing (e.g., 5x5 grid + small circle)
        centers = np.zeros((n, 2))
        r_base = 0.1
        k = 0
        for i in range(5):
            for j in range(5):
                centers[k] = [r_base + j * 2 * r_base, r_base + i * 2 * r_base]
                k += 1
        # Place 26th circle in a gap
        # Gap at center (0.5, 0.5) is occupied.
        # Gaps are at (0.1+0.1, 0.1+0.1) -> (0.2, 0.2) is center of a void?
        # Void centers are at (0.2, 0.2), (0.4, 0.2), etc.
        # Distance from (0.2, 0.2) to (0.1, 0.1) is sqrt(0.01+0.01)=0.1414.
        # Radius of void circle = 0.1414 - 0.1 = 0.0414.
        centers[25] = [0.2, 0.2]
        radii = np.full(n, r_base)
        radii[25] = 0.04
        
        # Verify fallback
        if not validate_packing(centers, radii):
             # If even fallback fails (unlikely), shrink slightly
             radii *= 0.99

    return centers, radii, np.sum(radii)

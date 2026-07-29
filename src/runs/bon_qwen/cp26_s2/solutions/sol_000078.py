# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3e28a2dc) state=d10a6236 sum of radii=0.294616 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def initialize_hex_grid(n=26):
    """Initialize circle centers in a hexagonal pattern."""
    pts = []
    s = 0.17  # Initial spacing
    for j in range(12):
        for i in range(12):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if x <= 1.0 and y <= 1.0:
                pts.append([x, y])
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
            
    pts = np.array(pts[:n])
    # Normalize to fit comfortably inside [0,1]x[0,1]
    min_val = pts.min(axis=0)
    max_val = pts.max(axis=0)
    range_val = max_val - min_val
    if np.any(range_val > 0):
        pts = (pts - min_val) / range_val
    pts = pts * 0.65 + 0.15  # Center and scale to leave room for growth
    return pts

def run_packing():
    np.random.seed(42)
    n = 26
    
    centers = initialize_hex_grid(n)
    radii = np.full(n, 0.01)
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = 0.0
    steps_no_improvement = 0
    
    dt = 0.006
    growth_factor = 1.00025
    
    for step in range(7000):
        fx = np.zeros(n)
        fy = np.zeros(n)
        max_ov = 0.0
        
        # Pairwise repulsion forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.hypot(dx, dy)
                req = radii[i] + radii[j]
                
                if dist < req:
                    ov = req - dist
                    if ov > max_ov:
                        max_ov = ov
                        
                    if dist > 1e-9:
                        nx = dx / dist
                        ny = dy / dist
                        # Strong repulsion proportional to overlap
                        f = ov * 150.0
                        fx[i] += nx * f
                        fy[i] += ny * f
                        fx[j] -= nx * f
                        fy[j] -= ny * f
                    else:
                        # Handle coincident points
                        fx[i] += 1.0
                        fy[i] += 1.0
                        fx[j] -= 1.0
                        fy[j] -= 1.0

        # Boundary repulsion forces
        for i in range(n):
            r = radii[i]
            if centers[i, 0] - r < 0:
                fx[i] += (r - centers[i, 0]) * 200.0
            if centers[i, 0] + r > 1:
                fx[i] -= (centers[i, 0] + r - 1) * 200.0
            if centers[i, 1] - r < 0:
                fy[i] += (r - centers[i, 1]) * 200.0
            if centers[i, 1] + r > 1:
                fy[i] -= (centers[i, 1] + r - 1) * 200.0
                
        # Update positions
        centers += dt * np.column_stack((fx, fy))
        
        # Grow radii if configuration is stable
        if max_ov < 5e-5:
            radii *= growth_factor
            
        # Decay parameters for convergence
        dt *= 0.9995
        growth_factor = 1.0 + (growth_factor - 1.0) * 0.998
        
        # Validate and track best
        is_valid = True
        # Boundary check
        for i in range(n):
            if (centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] + radii[i] > 1.0 + 1e-9 or
                centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] + radii[i] > 1.0 + 1e-9):
                is_valid = False
                break
                
        if is_valid:
            # Overlap check
            for i in range(n):
                for j in range(i + 1, n):
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-10:
                        is_valid = False
                        break
                if not is_valid:
                    break
                    
        if is_valid:
            curr_sum = np.sum(radii)
            if curr_sum > best_sum + 1e-9:
                best_sum = curr_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                steps_no_improvement = 0
            else:
                steps_no_improvement += 1
                
        # Escape local minima if stuck
        if steps_no_improvement > 600:
            perturbation = np.random.normal(0, 0.005, size=centers.shape)
            centers += perturbation
            steps_no_improvement = 0
            
    return best_centers, best_radii, best_sum

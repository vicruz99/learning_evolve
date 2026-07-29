# sol_000216 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b505a133) state=9f82a338 sum of radii=1.560780 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed optimization approach to find a dense packing.
    """
    n_circles = 26
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Initialize centers in a 6x5 grid pattern (30 spots, we use 26)
    # Starting with a structured layout helps avoid terrible local minima.
    centers = np.zeros((n_circles, 2))
    # Start with a radius that definitely fits (e.g., 0.06)
    # A 6x5 grid with spacing 1/5 and 1/6 allows radii up to ~0.08.
    radii = np.ones(n_circles) * 0.06 
    
    idx = 0
    # Create a 6 rows x 5 cols grid
    for r in range(6):
        for c in range(5):
            if idx >= n_circles:
                break
            # Uniform spacing
            x = (c + 0.5) / 5.0
            y = (r + 0.5) / 6.0
            # Add small random jitter to break symmetry and allow exploration
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            centers[idx] = [x, y]
            idx += 1
            
    # Optimization parameters
    lr = 0.15       # Learning rate for position updates (force scaling)
    growth_rate = 1.0005 # Factor to increase radii by each step
    max_steps = 1500     # Total number of growth iterations
    sub_steps = 8        # Number of force resolution iterations per growth step
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    current_centers = centers.copy()
    current_radii = radii.copy()
    
    for step in range(max_steps):
        # 1. Grow radii to push for higher density
        current_radii *= growth_rate
        
        # 2. Resolve overlaps using repulsion forces
        for _ in range(sub_steps):
            forces = np.zeros_like(current_centers)
            max_overlap = 0.0
            
            # Compute pairwise overlaps and boundary overlaps
            for i in range(n_circles):
                xi, yi = current_centers[i]
                ri = current_radii[i]
                
                # Boundary repulsion
                # Left wall
                if xi < ri:
                    ov = ri - xi
                    if ov > max_overlap: max_overlap = ov
                    forces[i, 0] += ov
                # Right wall
                if xi > 1.0 - ri:
                    ov = ri - (1.0 - xi)
                    if ov > max_overlap: max_overlap = ov
                    forces[i, 0] -= ov
                # Bottom wall
                if yi < ri:
                    ov = ri - yi
                    if ov > max_overlap: max_overlap = ov
                    forces[i, 1] += ov
                # Top wall
                if yi > 1.0 - ri:
                    ov = ri - (1.0 - yi)
                    if ov > max_overlap: max_overlap = ov
                    forces[i, 1] -= ov
                
                # Pairwise repulsion
                for j in range(i + 1, n_circles):
                    xj, yj = current_centers[j]
                    rj = current_radii[j]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist_sq = dx*dx + dy*dy
                    min_dist = ri + rj
                    
                    # Check for overlap
                    if dist_sq < min_dist * min_dist:
                        dist = math.sqrt(dist_sq)
                        if dist < 1e-9:
                            dist = 1e-9
                            dx = 0.01 # Avoid division by zero, random nudge
                            dy = 0.0
                            
                        overlap = min_dist - dist
                        if overlap > max_overlap:
                            max_overlap = overlap
                        
                        # Force magnitude proportional to overlap
                        # Direction is along the line connecting centers (push apart)
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                        
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Apply forces to move centers
            current_centers += forces * lr
            
            # Ensure centers stay within bounds (safety clamp)
            current_centers = np.clip(current_centers, 0.0, 1.0)
            
            # If overlaps are resolved, break early to save time
            if max_overlap < 1e-6:
                break
        
        # If significant overlap remains after resolution, shrink radii to recover validity
        if max_overlap > 0.01:
            current_radii /= 1.01 
        elif max_overlap > 1e-4:
            # Slight shrink for moderate overlap to ensure convergence
            current_radii /= 1.005
            
        # Strict validation check periodically to update best solution
        if step % 100 == 0:
            valid = True
            # Check boundaries
            for i in range(n_circles):
                x, y = current_centers[i]
                r = current_radii[i]
                if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
                    valid = False
                    break
                # Check pairwise overlaps
                for j in range(i + 1, n_circles):
                    dx = x - current_centers[j, 0]
                    dy = y - current_centers[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < r + current_radii[j] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_sum = np.sum(current_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
                    
        # Random perturbation to escape local minima
        if step % 200 == 0:
            current_centers += np.random.normal(0, 0.05, (n_circles, 2))
            current_centers = np.clip(current_centers, 0.0, 1.0)
            # Shrink radii slightly to accommodate the jitter and allow re-optimization
            current_radii *= 0.9

    return best_centers, best_radii, best_sum

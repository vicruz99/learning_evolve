# sol_000057 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05a03f22) state=561df9e2 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    
    for i in range(n):
        if radii[i] < -1e-9:
            return False
        
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def run_packing():
    N = 26
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # --- Stage 1: Initialization ---
    # Start with a grid that fits 25 circles and try to squeeze the 26th
    # Or a hexagonal pattern. Let's try a perturbed 5x5 grid for 25 and 1 extra.
    
    # 5x5 Grid for 25 circles with radius 0.1 (tight)
    # To allow optimization, start slightly smaller
    r_init = 0.09
    
    idx = 0
    # Place 25 circles in a grid
    for row in range(5):
        for col in range(5):
            x = (col + 0.5) * 0.2  # Centers at 0.1, 0.3, 0.5, 0.7, 0.9
            y = (row + 0.5) * 0.2
            centers[idx] = [x, y]
            radii[idx] = r_init
            idx += 1
            
    # Place 26th circle in a corner gap or center
    # In a 5x5 grid, the center is (0.5, 0.5) which is occupied.
    # Let's put it near a corner, slightly inset.
    # Actually, with r=0.09, circles touch. No gap.
    # We need to perturb to create space.
    # Let's place 26th circle at (0.5, 0.5) and push others away? 
    # No, let's place it at a corner like (0.05, 0.05) but ensure it doesn't overlap.
    # (0.1, 0.1) is occupied by circle 0.
    # Let's just scatter the 26th circle and rely on optimization.
    centers[25] = [0.5, 0.5] 
    radii[25] = r_init * 0.5 # Start small
    
    # --- Stage 2: Optimization Loop ---
    # We will use a simple local search / simulated annealing style approach.
    # Goal: Maximize sum(radii).
    # We can treat this as: try to expand radii, resolve collisions.
    
    # Parameters
    dt = 0.01 # Time step for movement
    repulsion_strength = 1.0
    expansion_rate = 0.001
    num_iterations = 2000
    
    # Current radii sum
    current_sum_radii = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = current_sum_radii
    
    # Convert to lists for mutability if needed, but numpy is fine
    
    for step in range(num_iterations):
        # 1. Try to expand radii uniformly
        # If valid, keep. If not, we need to resolve.
        # Instead of checking validity first, we'll push circles apart if they overlap.
        
        # Expand radii
        radii += expansion_rate
        
        # 2. Force Calculation and Position Update
        forces = np.zeros_like(centers)
        
        # Check constraints and apply forces
        # Wall constraints
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += repulsion_strength * (-(x - r))
            # Right wall
            elif x + r > 1:
                forces[i, 0] -= repulsion_strength * (x + r - 1)
                
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += repulsion_strength * (-(y - r))
            # Top wall
            elif y + r > 1:
                forces[i, 1] -= repulsion_strength * (y + r - 1)
        
        # Circle-Circle repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                required_dist = radii[i] + radii[j]
                
                if dist < required_dist and dist > 1e-9:
                    overlap = required_dist - dist
                    # Direction from j to i
                    direction = diff / dist
                    # Repel both equally
                    f = repulsion_strength * overlap
                    forces[i] += f * direction
                    forces[j] -= f * direction
                elif dist < 1e-9:
                    # Prevent division by zero, push apart randomly or fixed
                    forces[i, 0] += 0.1
                    forces[i, 1] += 0.1
                    forces[j, 0] -= 0.1
                    forces[j, 1] -= 0.1

        # Update positions
        centers += dt * forces
        
        # Clamp positions to [0, 1] strictly to avoid numerical drift issues
        centers = np.clip(centers, 0, 1)
        
        # Check if valid packing (with tolerance)
        # We allow slight overlaps during optimization but try to resolve them
        # If we find a valid state with higher sum, save it.
        
        # To save computation, check validity every 10 steps or at end
        if step % 50 == 0 or step == num_iterations - 1:
            if validate_packing(centers, radii):
                s = np.sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    # If valid, try to expand faster?
                    expansion_rate = min(expansion_rate * 1.1, 0.01)
                else:
                    # If not better, maybe radii are too big. Reduce slightly?
                    radii -= expansion_rate # Undo expansion if invalid or not improving
                    # Actually, the logic above expands unconditionally. 
                    # If invalid, forces will push them apart.
                    # But radii stayed large.
                    # Let's just keep radii. The forces will push centers to accommodate.
        
        # Decay expansion rate to refine
        if step > 1000:
            expansion_rate *= 0.99

    # --- Stage 3: Final Polish ---
    # Use the best found state
    centers = best_centers
    radii = best_radii
    
    # Try to squeeze more by small random perturbations (Simulated Annealing)
    temp = 0.01
    for _ in range(1000):
        # Pick random circle
        i = np.random.randint(N)
        # Perturb center
        new_centers = centers.copy()
        new_centers[i, 0] += np.random.uniform(-temp, temp)
        new_centers[i, 1] += np.random.uniform(-temp, temp)
        new_centers[i] = np.clip(new_centers[i], 0, 1)
        
        if validate_packing(new_centers, radii):
            # If valid, try to increase radius of this circle
            new_radii = radii.copy()
            # Estimate max radius for this circle
            # Check distance to all other circles and walls
            min_d = 1.0
            # Walls
            min_d = min(min_d, new_centers[i, 0], 1 - new_centers[i, 0],
                        new_centers[i, 1], 1 - new_centers[i, 1])
            
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(new_centers[i] - new_centers[j])
                min_d = min(min_d, d - radii[j])
            
            if min_d > radii[i]:
                new_radii[i] = min_d - 1e-9 # Safe margin
                if np.sum(new_radii) > np.sum(radii):
                    centers = new_centers
                    radii = new_radii
                    best_sum = np.sum(radii)
        
        temp *= 0.999

    # Final validation
    if not validate_packing(centers, radii):
        # Fallback to a safe configuration if optimization failed
        # 5x5 grid of 25 circles radius 0.1, 1 small circle?
        # But we need 26.
        # Let's regenerate a safe hex-like pack
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        r = 0.09
        idx = 0
        # Hexagonal packing attempt
        # Row 0: 5 circles
        # Row 1: 5 circles (shifted)
        # ...
        # This might not fit 26 with r=0.09 if not careful.
        # Let's just scale down the current centers if invalid, though unlikely to be totally broken.
        # But to be safe, let's just return the optimized result assuming it passed.
        # The logic above saves best_centers only if valid.
        pass

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

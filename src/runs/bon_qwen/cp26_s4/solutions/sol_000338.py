# sol_000338 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9bf69ab6) state=c48b1fea sum of radii=0.260000 correctness=1.0
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

def calculate_max_radius_for_centers(centers):
    """
    For fixed centers, calculates the maximum radius for each circle
    satisfying overlap and boundary constraints.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize radii based on boundary distance
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, 1 - x, y, 1 - y)
        
    # Iterate to satisfy overlap constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                max_r = dist / 2.0
                if radii[i] + radii[j] > max_r:
                    # Adjust radii
                    diff = (radii[i] + radii[j]) - max_r
                    radii[i] -= diff / 2
                    radii[j] -= diff / 2
                    changed = True
        if not changed:
            break
            
    return radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # 1. Hexagonal Grid Initialization
    centers = np.zeros((n, 2))
    idx = 0
    
    # Estimate grid parameters
    rows = 6
    cols = 4
    # We need 26 points. 6 rows of 4 or 5?
    # Let's try to fit 5 rows of 5 + 1? Or 6 rows of 4 + 2?
    # A staggered hexagonal layout:
    # Row 0: 4, Row 1: 5, Row 2: 4, Row 3: 5, Row 4: 4, Row 5: 4 -> Total 26
    
    # Layout definition
    layout = [4, 5, 4, 5, 4, 4] 
    total_count = sum(layout)
    
    if total_count != n:
        # Fallback to 5x5 + 1
        layout = [5, 5, 5, 5, 5, 1]
        
    # Normalize coordinates to [0, 1]
    # We will adjust spacing later
    row_height = 1.0 / (len(layout) + 1)
    for r_idx, count in enumerate(layout):
        y = (r_idx + 1) * row_height
        x_start = 0.5 / count # Rough centering
        # Evenly spaced in row
        if r_idx % 2 == 1: # Shifted row
            x_step = 1.0 / (count + 1)
            x_start = x_step
        else:
            x_step = 1.0 / (count + 1)
            x_start = x_step
            
        for c in range(count):
            centers[idx] = [x_start + c * x_step, y]
            idx += 1
            
    # Initial radii estimation
    initial_r = 1.0 / (2.0 * max(layout)) # Very rough guess
    radii = np.full(n, initial_r)
    
    # 2. Optimization (Simulated Annealing with Penalty)
    # Objective: Maximize sum(radii) - Penalty
    # Penalty = sum of squared overlap/boundary violations
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    temperature = 0.05 # High initial temp for large moves
    
    for step in range(5000):
        # Cool down
        temperature *= 0.995
        
        # Create a candidate
        cand_centers = best_centers.copy()
        cand_radii = best_radii.copy()
        
        # Perturb centers
        perturbation = np.random.normal(0, temperature, size=cand_centers.shape)
        cand_centers += perturbation
        cand_centers = np.clip(cand_centers, 1e-5, 1 - 1e-5)
        
        # Calculate feasible radii for these centers
        # This implicitly handles constraints
        cand_radii = calculate_max_radius_for_centers(cand_centers)
        
        cand_sum = np.sum(cand_radii)
        
        # Accept or Reject
        if cand_sum > best_sum:
            best_centers = cand_centers
            best_radii = cand_radii
            best_sum = cand_sum
        else:
            # Metropolis criterion
            diff = cand_sum - best_sum
            prob = np.exp(diff / (temperature + 1e-9))
            if np.random.random() < prob:
                best_centers = cand_centers
                best_radii = cand_radii
                best_sum = cand_sum
                
        # Periodic validation check (expensive, do occasionally)
        if step % 500 == 0 and step > 0:
            if validate_packing(best_centers, best_radii):
                pass # Valid
            else:
                # Reset to last known valid if drifted? 
                # With calculate_max_radius_for_centers, it should always be valid
                pass

    # 3. Final Polish
    # Ensure all radii are feasible
    best_radii = calculate_max_radius_for_centers(best_centers)
    
    # Final validation
    if not validate_packing(best_centers, best_radii):
        # Fallback to a valid small packing if optimization failed
        centers = np.random.rand(26, 2)
        radii = np.full(26, 0.01)
    else:
        centers = best_centers
        radii = best_radii
        
    final_sum = np.sum(radii)
    return centers, radii, final_sum

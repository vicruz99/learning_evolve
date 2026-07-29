# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f294fc76) state=d82c4837 sum of radii=1.569691 correctness=1.0
# stdout(first 200): Sum of radii: 1.5273834547356753 Validation: True
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from itertools import permutations

def validate_packing(centers, radii):
    """
    Validation function (read-only)
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0: return False
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

def compute_optimal_radii(centers):
    """
    For a fixed set of centers, compute the maximum possible radii 
    such that circles do not overlap and stay inside the unit square.
    This is solved iteratively as a water-filling problem on the conflict graph.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initial bounds based on distance to boundaries
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, 1 - x, y, 1 - y)

    # Iteratively adjust radii based on neighbor constraints
    # r_i + r_j <= dist(i, j)
    # This converges quickly for small N
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    # Iterate until convergence
    for _ in range(50):
        max_change = 0
        for i in range(n):
            current_r = radii[i]
            # Check boundary
            limit = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            # Check neighbors
            for j in range(n):
                if i != j:
                    limit = min(limit, dist_matrix[i, j] - radii[j])
            
            # Ensure non-negative
            limit = max(0, limit)
            
            if abs(limit - current_r) > 1e-9:
                radii[i] = limit
                max_change = max(max_change, abs(limit - current_r))
        
        if max_change < 1e-9:
            break
            
    return radii

def calculate_score(centers):
    radii = compute_optimal_radii(centers)
    return np.sum(radii), radii

def generate_initial_centers(n):
    """
    Generate initial centers using a hexagonal lattice pattern.
    """
    centers = np.zeros((n, 2))
    
    # Parameters for hexagonal packing
    # We want to fit n circles. 
    # Approximate grid size
    cols = math.ceil(math.sqrt(n * 2 / math.sqrt(3)))
    rows = math.ceil(n / cols)
    
    # Adjust dimensions to fit in unit square
    # Hexagonal spacing: horizontal 2r, vertical sqrt(3)r
    # Let's assume a radius r ~ 0.1. Spacing ~ 0.2 and 0.1732
    # We scale to fill the square
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            # Hexagonal coordinates
            x = col * 0.2 + (row % 2) * 0.1
            y = row * 0.1732
            
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            
    # Scale and center to unit square
    # Find bounds
    min_x, max_x = centers[:, 0].min(), centers[:, 0].max()
    min_y, max_y = centers[:, 1].min(), centers[:, 1].max()
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 or height == 0:
        # Fallback to grid
        side = math.sqrt(n)
        for i in range(n):
            centers[i, 0] = (i % side) / side
            centers[i, 1] = (i // side) / side
        return centers

    # Scale to fit in [0, 1] with some margin
    scale = 0.9 / max(width, height)
    centers[:, 0] = (centers[:, 0] - min_x) * scale + 0.05
    centers[:, 1] = (centers[:, 1] - min_y) * scale + 0.05
    
    return centers

def run_packing():
    """
    Returns (centers, radii, sum_radii) for the optimal packing of 26 circles.
    """
    n = 26
    best_centers = None
    best_sum = 0.0
    best_radii = None
    
    # Run multiple iterations with different initial permutations
    for trial in range(5):
        # Generate initial configuration
        centers = generate_initial_centers(n)
        
        # Add some random perturbation
        centers += np.random.uniform(-0.05, 0.05, size=centers.shape)
        # Clip to valid range (with padding)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Local optimization: Hill climbing with simulated annealing-like moves
        current_score, current_radii = calculate_score(centers)
        best_trial_score = current_score
        
        temperature = 0.1
        decay = 0.95
        
        for step in range(200):
            # Pick a random circle to move
            i = np.random.randint(n)
            old_pos = centers[i].copy()
            
            # Move it slightly
            delta = np.random.normal(0, temperature, size=2)
            new_pos = old_pos + delta
            
            # Check boundary constraints (keep it inside with some margin for radius)
            # We don't know radius yet, but keep it in [0.01, 0.99]
            if not (0.01 < new_pos[0] < 0.99 and 0.01 < new_pos[1] < 0.99):
                continue
                
            centers[i] = new_pos
            new_score, _ = calculate_score(centers)
            
            # Accept or reject
            if new_score > best_trial_score:
                best_trial_score = new_score
                # Keep move
            else:
                # Accept with probability exp((new - old)/T)
                if np.random.rand() < math.exp((new_score - current_score) / temperature):
                    current_score = new_score
                    # Keep move
                else:
                    centers[i] = old_pos # Revert
            
            temperature *= decay
            current_score, _ = calculate_score(centers)
            
            if best_trial_score > best_sum:
                best_sum = best_trial_score
                best_centers = centers.copy()
                best_radii = compute_optimal_radii(centers).copy()

    # Final validation and adjustment
    if best_centers is None:
        # Fallback to grid
        side = math.sqrt(n)
        best_centers = np.zeros((n, 2))
        for i in range(n):
            best_centers[i, 0] = (i % side) / side
            best_centers[i, 1] = (i // side) / side
        best_radii = compute_optimal_radii(best_centers)
        best_sum = np.sum(best_radii)

    # Final refinement of best_centers
    final_radii = compute_optimal_radii(best_centers)
    final_sum = np.sum(final_radii)
    
    return best_centers, final_radii, final_sum

# Execute the packing function
centers, radii, sum_radii = run_packing()

# Print results for verification
print(f"Sum of radii: {sum_radii}")
print(f"Validation: {validate_packing(centers, radii)}")

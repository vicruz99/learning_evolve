# sol_000182 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4cc484ef) state=8748adef sum of radii=2.104666 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < -1e-12:
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Seed for reproducibility or exploration
    np.random.seed(42)
    
    # Try multiple restarts to find the best local optimum
    for restart in range(50):
        # Initialize centers and radii
        centers = np.random.rand(n, 2)
        radii = np.ones(n) * 0.01 # Start small to ensure validity
        
        # Simple optimization loop
        learning_rate = 0.005
        alpha = 1.0
        
        for step in range(1000):
            force_centers = np.zeros_like(centers)
            force_radii = np.zeros(n)
            
            # 1. Expand Radii (Objective gradient)
            # We want to increase radii as much as possible
            force_radii += 1.0 

            # 2. Check Boundary Constraints
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                
                # Distance to boundaries
                dist_left = x
                dist_right = 1.0 - x
                dist_down = y
                dist_up = 1.0 - y
                min_dist_boundary = min(dist_left, dist_right, dist_down, dist_up)
                
                if r > min_dist_boundary - 1e-4:
                    # Penalty: shrink radius if too big
                    force_radii[i] -= (r - min_dist_boundary) * 10.0
                    # Push center away from boundary
                    if dist_left == min_dist_boundary:
                        force_centers[i, 0] += 10.0
                    elif dist_right == min_dist_boundary:
                        force_centers[i, 0] -= 10.0
                    elif dist_down == min_dist_boundary:
                        force_centers[i, 1] += 10.0
                    elif dist_up == min_dist_boundary:
                        force_centers[i, 1] -= 10.0

            # 3. Check Overlap Constraints
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9: dist = 1e-9 # Avoid division by zero
                    
                    r_sum = radii[i] + radii[j]
                    overlap = r_sum - dist
                    
                    if overlap > 0:
                        # Repulsive force
                        repulsion = overlap * 5.0
                        
                        # Force on radii (shrink)
                        force_radii[i] -= repulsion
                        force_radii[j] -= repulsion
                        
                        # Force on centers (push apart)
                        fx = (dx / dist) * repulsion
                        fy = (dy / dist) * repulsion
                        force_centers[i, 0] += fx
                        force_centers[i, 1] += fy
                        force_centers[j, 0] -= fx
                        force_centers[j, 1] -= fy

            # Update Centers
            centers += learning_rate * alpha * force_centers
            # Clamp centers to [0, 1]
            centers = np.clip(centers, 0.0, 1.0)
            
            # Update Radii
            radii += learning_rate * alpha * force_radii
            # Clamp radii to be non-negative
            radii = np.maximum(radii, 1e-6)
            
            # Adjust learning rate slightly
            if step % 100 == 0 and step > 0:
                alpha *= 0.95

            # Check validity and update best
            # We do a strict check occasionally or rely on the forces
            # For the final output, we must ensure strict validity.
            # The forces above are designed to keep it valid or correct violations.
            
        # Post-process: Clean up minor overlaps due to float precision
        # A simple relaxation step to ensure valid packing
        for k in range(50):
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                # Boundary check
                r_boundary = min(x, 1-x, y, 1-y)
                if r > r_boundary:
                    radii[i] = r_boundary
                    
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9: dist = 1e-9
                    r_sum = radii[i] + radii[j]
                    if dist < r_sum:
                        # Reduce radii to fit
                        scale = dist / r_sum
                        radii[i] *= scale
                        radii[j] *= scale

        current_sum = np.sum(radii)
        if current_sum > best_sum:
            # Validate before saving
            if validate_packing(centers, radii):
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    return best_centers, best_radii, best_sum

if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    print(centers)
    print(radii)
    print("Valid:", validate_packing(centers, radii))

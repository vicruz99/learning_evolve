# sol_000226 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f043a2e3) state=bbb8ac4c sum of radii=0.260000 correctness=1.0
# stdout(first 200): Packing valid: True Sum of radii: 0.26 Target: 2.636 Gap: 2.3760000000000003
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def generate_hexagonal_centers(n, seed=0):
    """
    Generates an initial configuration of n centers using a hexagonal grid 
    with a repulsion force to spread them out within the unit square.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize centers roughly uniformly
    centers = rng.uniform(0.1, 0.9, size=(n, 2))
    
    # Repulsion-based relaxation to spread points
    # This helps avoid clusters and creates a more uniform distribution
    for _ in range(500):
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 0.25 and dist > 1e-6:
                    # Strong repulsion force inversely proportional to distance squared
                    force = diff / (dist**2 + 1e-5) 
                    forces[i] += force
                    forces[j] -= force
            
            # Boundary repulsion (keep points away from edges)
            margin = 0.05
            if centers[i, 0] < margin: forces[i, 0] += 10 * (margin - centers[i, 0])
            if centers[i, 0] > 1 - margin: forces[i, 0] -= 10 * (centers[i, 0] - (1 - margin))
            if centers[i, 1] < margin: forces[i, 1] += 10 * (margin - centers[i, 1])
            if centers[i, 1] > 1 - margin: forces[i, 1] -= 10 * (centers[i, 1] - (1 - margin))
        
        # Update positions
        centers += 0.005 * forces
        # Clip to valid range
        centers = np.clip(centers, 0.01, 0.99)
        
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26,) with radius of each circle
        sum_radii: float, the sum of radii
    """
    n = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # We try a few different seeds to escape local minima
    seeds = [0, 42, 123, 777, 9999]
    
    for seed in seeds:
        # 1. Initialize centers
        centers = generate_hexagonal_centers(n, seed=seed)
        
        # 2. Initialize radii (small but positive)
        radii = np.full(n, 0.05)
        
        # 3. Prepare variables for optimization
        # Vector format: [x1, y1, ..., x26, y26, r1, ..., r26]
        x0 = np.concatenate([centers.flatten(), radii])
        
        # 4. Define bounds
        # x, y in [0, 1], r >= 0 (upper bound for r is 0.5)
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        # 5. Define objective function (with penalties)
        # We maximize sum(radii) => minimize -sum(radii)
        def objective(vars_flat):
            c = vars_flat[:2 * n].reshape(n, 2)
            r = vars_flat[2 * n:]
            
            # Objective
            obj = -np.sum(r)
            
            # Penalty term
            pen = 0.0
            
            # Boundary penalties (squared to be smooth and penalize hard violations)
            # x - r >= 0
            pen += np.sum(np.maximum(0, r - c[:, 0])**2)
            # x + r <= 1
            pen += np.sum(np.maximum(0, c[:, 0] + r - 1)**2)
            # y - r >= 0
            pen += np.sum(np.maximum(0, r - c[:, 1])**2)
            # y + r <= 1
            pen += np.sum(np.maximum(0, c[:, 1] + r - 1)**2)
            
            # Overlap penalties
            # dist(i, j) >= r_i + r_j
            # Compute pairwise distances
            # Using broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            
            rad_sums = r[:, np.newaxis] + r[np.newaxis, :]
            
            # Overlap violation: max(0, r_i + r_j - dist)
            # We only care about i < j to avoid double counting, but summing all is fine too
            violations = np.maximum(0, rad_sums - dists)
            pen += 2000.0 * np.sum(violations**2)
            
            return obj + 1000.0 * pen

        # 6. Optimize
        # L-BFGS-B is efficient for bound-constrained problems
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-10})
        
        # 7. Extract results
        current_centers = res.x[:2 * n].reshape(n, 2)
        current_radii = res.x[2 * n:]
        current_sum = np.sum(current_radii)
        
        # 8. Validate and keep best valid
        # We use the provided validation logic (simplified here)
        valid = True
        # Check boundaries
        if np.any(current_centers[:, 0] - current_radii < -1e-7) or \
           np.any(current_centers[:, 0] + current_radii > 1 + 1e-7) or \
           np.any(current_centers[:, 1] - current_radii < -1e-7) or \
           np.any(current_centers[:, 1] + current_radii > 1 + 1e-7):
            valid = False
            
        # Check overlaps
        if valid:
            diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            rad_sums = current_radii[:, np.newaxis] + current_radii[np.newaxis, :]
            # Check strictly for overlap
            if np.any(dists < rad_sums - 1e-7):
                valid = False

        if valid and current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()

    # If no valid packing found (unlikely), return a fallback
    if best_centers is None:
        fallback_centers = np.random.rand(26, 2) * 0.8 + 0.1
        fallback_radii = np.full(26, 0.01)
        return fallback_centers, fallback_radii, np.sum(fallback_radii)

    return best_centers, best_radii, best_sum_radii

# Helper to validate using the logic from the prompt (for internal check)
def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
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

# Run the packing to get the result
centers, radii, sum_radii = run_packing()

# Final validation
is_valid = validate_packing(centers, radii)
print(f"Packing valid: {is_valid}")
print(f"Sum of radii: {sum_radii}")
print(f"Target: 2.636")
print(f"Gap: {2.636 - sum_radii}")

# Return the result in the requested format for the function
# The problem asks to define run_packing, which we did.
# But to be safe with the execution environment, we can just rely on the function definition.
# However, usually these prompts imply running the code to generate output or just defining it.
# The prompt says "Make sure to /think step by step, first give your strategy... then finally return the final program".
# The program should contain the run_packing function.

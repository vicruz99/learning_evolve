# sol_000111 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 82d73ba2) state=b2e13550 sum of radii=2.484761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize, differential_evolution
import random

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def get_max_radii(centers):
    """
    Given centers, compute the maximum possible radius for each circle
    such that they don't overlap and stay inside the unit square.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    for i in range(n):
        # Distance to boundaries
        x, y = centers[i]
        dist_boundary = min(x, 1-x, y, 1-y)
        
        # Distance to other centers
        dist_centers = np.inf
        if n > 1:
            # Vectorized distance calculation
            diffs = centers - centers[i]
            # Zero out self-distance
            np.fill_diagonal(np.zeros((n, n)), 0) # Not needed if we handle indices
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf # Ignore self
            dist_centers = np.min(dists)
            
        # The radius is limited by half the distance to the nearest neighbor
        # and the distance to the boundary.
        radii[i] = min(dist_boundary, dist_centers / 2.0)
        
    return radii

def objective_negative_sum_radii(centers_flat):
    """
    Objective function to minimize (negative sum of radii).
    """
    n = 26
    centers = centers_flat.reshape((n, 2))
    radii = get_max_radii(centers)
    return -np.sum(radii)

def objective_with_penalty(centers_flat, penalty_factor=1000.0):
    """
    Objective function with penalties for invalid configurations (though get_max_radii handles validity by shrinking).
    Actually, get_max_radii always returns a valid packing.
    However, if we treat radii as variables, we need constraints.
    Here we just optimize centers.
    """
    return objective_negative_sum_radii(centers_flat)

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Strategy: Try multiple initial configurations and optimize
    # 1. Perturbed 5x5 grid + 1
    # 2. Random initializations
    
    def optimize_from_initial(initial_centers):
        # Use L-BFGS-B or similar. Since boundaries are [0,1], we can use bounds.
        bounds = [(0.0, 1.0)] * (2 * n)
        
        # Try a few local optimization runs with different method settings or just one robust one
        # Nelder-Mead is good for non-smooth functions
        try:
            res = minimize(objective_negative_sum_radii, initial_centers.flatten(), 
                           method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            return res.x, -res.fun
        except Exception:
            return initial_centers.flatten(), 0.0

    # Configuration 1: 5x5 Grid + 1
    # 5x5 grid centers
    grid_centers = []
    for i in range(5):
        for j in range(5):
            grid_centers.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    # Add 26th circle in a gap, e.g., (0.2, 0.2)
    # Actually (0.2, 0.2) is distance 0.141 from (0.1, 0.1).
    # Radius would be limited.
    grid_centers.append([0.2, 0.2])
    
    # Perturb slightly
    rng = np.random.default_rng(42)
    initial_1 = np.array(grid_centers) + rng.normal(0, 0.01, (26, 2))
    # Clamp to [0,1]
    initial_1 = np.clip(initial_1, 0.0, 1.0)
    
    # Configuration 2: Hexagonal packing attempt
    hex_centers = []
    # Rows
    r_est = 0.1 # Estimate
    y = r_est
    row_idx = 0
    while len(hex_centers) < 26 and y + r_est <= 1.0:
        x = r_est
        offset = (0.5 * r_est) if row_idx % 2 == 1 else 0.0
        x = r_est + offset
        while x + r_est <= 1.0:
            if len(hex_centers) < 26:
                hex_centers.append([x, y])
            x += 2 * r_est
        y += math.sqrt(3) * r_est
        row_idx += 1
    
    # Fill remaining with random if needed (shouldn't be)
    while len(hex_centers) < 26:
        hex_centers.append([rng.random(), rng.random()])
    initial_2 = np.array(hex_centers[:26])
    initial_2 = np.clip(initial_2, 0.0, 1.0)

    # Configuration 3: Random
    initial_3 = rng.random((26, 2))

    candidates = [initial_1, initial_2, initial_3]
    
    # Add some more random perturbations of grid
    for _ in range(5):
        init = np.array(grid_centers) + rng.normal(0, 0.05, (26, 2))
        init = np.clip(init, 0.0, 1.0)
        candidates.append(init)

    for i, init_centers in enumerate(candidates):
        centers_opt, current_sum = optimize_from_initial(init_centers)
        centers_opt = centers_opt.reshape((n, 2))
        radii_opt = get_max_radii(centers_opt)
        s = np.sum(radii_opt)
        
        # Verify validity explicitly
        if validate_packing(centers_opt, radii_opt):
            if s > best_sum:
                best_sum = s
                best_centers = centers_opt.copy()
                best_radii = radii_opt.copy()
        else:
            # If validation fails (due to numerical issues), try to fix by reducing radii slightly
            # get_max_radii should theoretically produce valid radii, but floating point errors might exist.
            # Let's force validity by clamping.
            # Re-calculate radii ensuring strict non-overlap
            # Actually get_max_radii uses min(dist/2, boundary).
            # dist/2 ensures dist >= r_i + r_j.
            # So it should be valid.
            pass

    # If best_sum is still low, try a more aggressive search or differential evolution on a subset
    # But 26 variables is a lot for DE.
    # Let's try one more local refinement on the best solution found.
    if best_centers is not None:
        # Refine
        try:
            res = minimize(objective_negative_sum_radii, best_centers.flatten(), 
                           method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            refined_centers = res.x.reshape((n, 2))
            refined_radii = get_max_radii(refined_centers)
            refined_sum = np.sum(refined_radii)
            
            if validate_packing(refined_centers, refined_radii):
                if refined_sum > best_sum:
                    best_sum = refined_sum
                    best_centers = refined_centers.copy()
                    best_radii = refined_radii.copy()
        except:
            pass

    return best_centers, best_radii, float(best_sum)

# To test locally
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")

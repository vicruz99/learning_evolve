# sol_000284 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0a16b6e7) state=747802a2 sum of radii=2.007290 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Helper to generate hexagonal initial guess
    def get_initial_guess(seed=0):
        rng = np.random.RandomState(seed)
        # Hexagonal packing approximation
        # Try to fit 26 circles. 5 rows of 5, 6, 5, 6, 4? Or 6, 5, 5, 5, 5?
        # Let's try a dense cluster.
        # Rows with counts: 5, 6, 5, 6, 4 -> Sum 26. 
        # Or simpler: 5 rows, varying lengths.
        # Let's just place them in a grid and perturb.
        
        # Better initial guess: Hexagonal lattice
        # Rows: 6, 5, 6, 5, 4 (Sum 26)
        # Row counts
        row_counts = [6, 5, 6, 5, 4]
        
        centers = []
        y_pos = 0.1 # Initial y
        spacing_y = 0.8 / 4 # Spread over height
        
        for i, count in enumerate(row_counts):
            # Centered x positions
            # Width available 1. 
            # If count=6, spacing roughly 1/6? 
            # Let's just spread them evenly
            xs = np.linspace(1.0/(2*count) * 1.1, 1.0 - 1.0/(2*count) * 1.1, count)
            # Actually, for initial guess, just uniform distribution in [0,1]
            xs = np.linspace(0.05, 0.95, count)
            
            # Shift odd rows for hexagonal pattern
            if i % 2 == 1:
                shift = 0.5 / count
                xs = xs + shift
                xs = np.clip(xs, 0.05, 0.95)

            for x in xs:
                centers.append([x, y_pos])
            
            y_pos += spacing_y

        # If we don't have enough, or too many, adjust. 
        # The above logic is a bit rough. Let's use a more robust random hex seed.
        centers = []
        # Try to pack in a hexagonal pattern
        # Approx radius 0.1
        r_est = 0.1
        # 6 circles per row approx.
        # 5 rows.
        
        # Let's just generate random positions and run optimization?
        # No, good initial guess is crucial.
        
        # Let's create a 5x6 grid (30 points) and remove 4, or 5x5 grid (25) + 1?
        # 5x5 grid centers:
        grid_x = np.linspace(0.1, 0.9, 5)
        grid_y = np.linspace(0.1, 0.9, 5)
        cx, cy = np.meshgrid(grid_x, grid_y)
        points = np.column_stack((cx.ravel(), cy.ravel()))
        
        # We need 26. Add one in the middle?
        # The 5x5 grid has 25 points.
        # Add a point at center (0.5, 0.5) - but that's occupied.
        # Perturb the grid to make space?
        # Actually, just take the first 26 points of a denser grid?
        # 6x5 grid has 30 points.
        
        grid_x = np.linspace(0.1, 0.9, 6) # 6 points
        grid_y = np.linspace(0.1, 0.9, 5) # 5 points
        cx, cy = np.meshgrid(grid_x, grid_y)
        points = np.column_stack((cx.ravel(), cy.ravel()))
        
        # Take first 26
        points = points[:26]
        
        # Add small random noise
        points += rng.uniform(-0.01, 0.01, points.shape)
        
        return points

    # Objective function with penalty
    def objective(vars):
        centers = vars[:2*n].reshape((n, 2))
        radii = vars[2*n:]
        
        # Penalty for negative radii
        pen = 0.0
        pen += np.sum(np.maximum(0, -radii))
        
        # Boundary penalties
        # x - r >= 0 -> r - x <= 0
        pen += np.sum(np.maximum(0, radii - centers[:, 0]))
        # x + r <= 1 -> x + r - 1 <= 0
        pen += np.sum(np.maximum(0, centers[:, 0] + radii - 1))
        # y - r >= 0
        pen += np.sum(np.maximum(0, radii - centers[:, 1]))
        # y + r <= 1
        pen += np.sum(np.maximum(0, centers[:, 1] + radii - 1))
        
        # Overlap penalties
        # dist >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (r_i + r_j)^2 - dist^2 <= 0
        # We penalize violations.
        
        # Vectorized overlap check
        # Compute pairwise distances squared
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        # violation = (r_sum)**2 - dist_sq
        # We only care about i < j
        # mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # pen += 100.0 * np.sum(np.maximum(0, violation[mask]))
        
        # Doing it in loops for clarity and lower memory if n is small, 
        # but vectorized is fine for 26.
        
        # To avoid huge penalty numbers, scale carefully.
        # But for optimization, large penalty is needed.
        
        # Let's use a simpler loop for overlap to be safe
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d2 = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                if d2 < r_sum*r_sum:
                    pen += 100.0 * (r_sum*r_sum - d2)
        
        return -np.sum(radii) + pen

    best_obj_val = np.inf

    # Run optimization multiple times with different seeds
    for seed in range(10):
        initial_centers = get_initial_guess(seed)
        initial_radii = np.full(n, 0.05) # Start small
        
        vars0 = np.concatenate([initial_centers.flatten(), initial_radii])
        
        # Bounds
        bounds = [(0, 1) for _ in range(2*n)] # x, y
        bounds += [(0, 0.5) for _ in range(n)] # r (max 0.5)

        res = minimize(objective, vars0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.fun < best_obj_val:
            best_obj_val = res.fun
            best_centers = res.x[:2*n].reshape((n, 2))
            best_radii = res.x[2*n:]

    # Post-processing: Refine using exact constraints if possible or just return best
    # The penalty method might allow slight violations. 
    # Let's clip radii to satisfy constraints strictly if needed, but the penalty was high.
    # Actually, let's re-validate and shrink radii slightly to be safe.
    
    # Safety check and adjustment
    # Check overlaps and reduce radii if necessary
    changed = True
    while changed:
        changed = False
        for i in range(n):
            # Boundary
            min_r = min(best_centers[i, 0], 1 - best_centers[i, 0], 
                        best_centers[i, 1], 1 - best_centers[i, 1])
            if min_r < best_radii[i] - 1e-9:
                best_radii[i] = max(0, min_r)
                changed = True
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                dist = math.hypot(dx, dy)
                req_dist = best_radii[i] + best_radii[j]
                if dist < req_dist - 1e-9:
                    # Reduce both radii equally to maintain sum? Or just reduce larger?
                    # Simplest: reduce both by half the violation
                    diff = req_dist - dist
                    best_radii[i] -= diff / 2
                    best_radii[j] -= diff / 2
                    best_radii[i] = max(0, best_radii[i])
                    best_radii[j] = max(0, best_radii[j])
                    changed = True

    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii

# Validation code (read-only, do not modify)
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

if __name__ == "__main__":
    centers, radii, s = run_packing()
    valid = validate_packing(centers, radii)
    print(f"Valid: {valid}")
    print(f"Sum of radii: {s}")

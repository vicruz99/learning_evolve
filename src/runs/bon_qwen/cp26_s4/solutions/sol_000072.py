# sol_000072 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a76a0e24) state=b8b284f5 sum of radii=2.032748 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Weight for the penalty term. 
    # High enough to enforce constraints, but not so high that it dominates the objective gradient.
    PENALTY_WEIGHT = 1000.0

    def objective(params):
        """
        Objective function: Minimize -sum(radii) + penalty for violations.
        params: array of shape (n*3,) containing [x1, y1, r1, x2, y2, r2, ...]
        """
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        # Initialize penalty
        penalty = 0.0
        
        # 1. Boundary constraints
        # Circle i must satisfy: r_i <= x_i <= 1 - r_i  => x_i - r_i >= 0 and 1 - x_i - r_i >= 0
        # Violation: max(0, r_i - x_i) and max(0, r_i - (1 - x_i))
        # We use squared penalty for smoothness (mostly)
        
        # x boundaries
        # r - x < 0 => x < r (left wall violation)
        # r - (1-x) < 0 => x + r > 1 (right wall violation)
        # Actually, constraint is x >= r and x <= 1-r.
        # So violation is max(0, r - x) + max(0, r - (1-x)) ?
        # No, if x < r, violation is r - x. If x > 1-r, violation is x - (1-r) = x + r - 1.
        
        # Let's compute distance to boundary for each circle
        # dist to left wall: x
        # dist to right wall: 1-x
        # dist to bottom wall: y
        # dist to top wall: 1-y
        # Constraint: r <= dist_to_wall
        # Violation: max(0, r - dist_to_wall)
        
        dists_to_walls = np.column_stack([
            centers[:, 0],              # x
            1 - centers[:, 0],          # 1-x
            centers[:, 1],              # y
            1 - centers[:, 1]           # 1-y
        ])
        
        # Violations where radius exceeds distance to wall
        wall_violations = np.maximum(0, radii[:, np.newaxis] - dists_to_walls)
        penalty += np.sum(wall_violations**2)
        
        # 2. Overlap constraints
        # dist(i, j) >= r_i + r_j
        # Violation: max(0, r_i + r_j - dist(i, j))
        
        # Compute pairwise distances efficiently
        # centers shape (n, 2)
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Sum of radii matrix
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount: positive if overlapping
        overlap = r_sum - dists
        
        # Zero out diagonal (distance to self is 0, r+r = 2r, but self doesn't overlap with self)
        np.fill_diagonal(overlap, -1e9) 
        
        # Positive overlaps contribute to penalty
        pos_overlap = np.maximum(0, overlap)
        
        # Sum of squared overlaps. 
        # Since matrix is symmetric, this sums each pair twice.
        # That's fine, just scales the penalty.
        penalty += np.sum(pos_overlap**2)
        
        # Objective: maximize sum(radii) -> minimize -sum(radii)
        # We add weighted penalty
        return -np.sum(radii) + PENALTY_WEIGHT * penalty

    # Define bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (radius cannot exceed 0.5 in unit square)
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Helper to generate initial guess
    def get_initial_guess(strategy='grid', seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        if strategy == 'grid':
            # Hexagonal-like grid
            # Try to fit 26 circles.
            # 6 columns, 5 rows would be 30.
            # Let's use a grid and pick 26.
            # Or better, a specific pattern.
            # Let's try a 5x5 grid (25) + 1 in center?
            # Or 6x5 grid.
            
            # Let's try to distribute them somewhat evenly
            # Sqrt(26) approx 5.1
            # Let's try 6 columns, 4 rows (24) + 2?
            # Or just random dense packing.
            
            # A good starting point is a hexagonal packing pattern.
            # Rows with offset.
            # Let's generate points on a hexagonal lattice and clip/scale to unit square.
            
            # Generate a large enough grid and sample 26 points
            # Or simply place them on a grid.
            
            # Let's try 6x5 grid
            xs = np.linspace(0.1, 0.9, 6)
            ys = np.linspace(0.1, 0.9, 5)
            grid_x, grid_y = np.meshgrid(xs, ys)
            centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            
            # Take first 26
            centers = centers[:n]
            
            # Small initial radii
            radii = np.full(n, 0.05)
            
        elif strategy == 'random':
            centers = np.random.rand(n, 2)
            radii = np.random.rand(n) * 0.05 # Small random radii
            
        return np.hstack([centers.ravel(), radii])

    best_params = None
    best_score = np.inf # Minimizing score
    best_sum_radii = -np.inf
    
    # Try multiple initializations
    strategies = ['grid', 'grid', 'random', 'random', 'random']
    
    for i, strat in enumerate(strategies):
        # Add some noise to grid to break symmetry
        if strat == 'grid':
            params = get_initial_guess('grid')
            # Perturb
            noise = np.random.normal(0, 0.02, params.shape)
            params = params + noise
            # Clip bounds
            params[0::3] = np.clip(params[0::3], 0.0, 1.0)
            params[1::3] = np.clip(params[1::3], 0.0, 1.0)
            params[2::3] = np.clip(params[2::3], 0.0, 0.5)
        else:
            params = get_initial_guess('random')

        try:
            res = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Extract result
            centers_opt = res.x[:2*n].reshape(n, 2)
            radii_opt = res.x[2*n:]
            
            # Check validity of the solution found by optimizer
            # The optimizer minimizes -sum(r) + W*pen.
            # If pen is low, it's good.
            
            # Compute actual sum of radii
            current_sum = np.sum(radii_opt)
            
            # Verify constraints strictly
            # We need to ensure no overlaps.
            # Due to numerical errors, there might be tiny overlaps.
            # We can check and if necessary, shrink radii slightly.
            # But let's first check the penalty value.
            # Re-evaluate penalty manually to be sure
            diff = centers_opt[:, np.newaxis, :] - centers_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            r_sum = radii_opt[:, np.newaxis] + radii_opt[np.newaxis, :]
            overlap = r_sum - dists
            np.fill_diagonal(overlap, 0)
            max_ov = np.max(overlap)
            
            wall_ov = 0.0
            for k in range(n):
                x, y = centers_opt[k]
                r = radii_opt[k]
                wall_ov = max(wall_ov, max(0, r-x), max(0, r-(1-x)), max(0, r-y), max(0, r-(1-y)))
                
            # If max overlap is very small (numerical noise), it's acceptable
            # But if it's significant, we might need to adjust.
            # However, L-BFGS-B with high penalty usually finds valid or near-valid solutions.
            
            # If we have a valid solution (or nearly valid), compare sum
            # We want to maximize sum_radii.
            # The optimizer minimizes -sum + penalty.
            # A valid solution has penalty ~0, so score ~ -sum.
            # Lower score is better.
            
            # Let's verify validity strictly for the "best" candidate
            # We will store candidates and validate at the end.
            
            if res.success or (max_ov < 1e-7 and wall_ov < 1e-7):
                # Valid enough
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_params = res.x.copy()
            else:
                # Even if not strictly valid in check, if penalty is low, keep it
                # The penalty in objective function is scaled by W.
                # If objective value is close to -sum_radii, it's good.
                # But simpler: just track max sum_radii among valid-ish solutions.
                # Let's just store all and pick best valid at end.
                pass

        except Exception as e:
            print(f"Optimization failed: {e}")

    # Fallback if best_params is None (should not happen)
    if best_params is None:
        params = get_initial_guess('grid')
        res = minimize(objective, params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        best_params = res.x

    centers = best_params[:2*n].reshape(n, 2)
    radii = best_params[2*n:]
    
    # Final validation and adjustment
    # Ensure no overlaps by shrinking radii if necessary
    # This is a safety step.
    # We can compute the "tightest" radius for each circle given others.
    # But simply checking and scaling down globally might be safer but reduces sum.
    # Instead, let's check pairwise and fix.
    
    # Check overlaps
    valid = True
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
            # Clip radius to fit in square
            max_r_boundary = min(x, 1-x, y, 1-y)
            if radii[i] > max_r_boundary + 1e-9:
                radii[i] = max_r_boundary
                valid = False # Was invalid, now fixed
            
    # Check pairwise overlaps
    # If overlap, reduce radius of one or both?
    # Reducing radii reduces sum.
    # But if the optimizer found a local min with overlap, we must fix it.
    # However, with high penalty, overlaps should be minimal.
    # Let's just ensure strict validity for the return.
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = dists[i, j]
            sum_r = radii[i] + radii[j]
            if dist < sum_r - 1e-9:
                # Overlap detected.
                # Reduce radii to just touch.
                # Distribute reduction? Or reduce larger one?
                # To maximize sum, we should reduce as little as possible.
                # Actually, if we are stuck in a configuration with overlap,
                # we might need to move centers.
                # But for the sake of returning a valid result:
                # We can scale down radii uniformly? No.
                # Let's just reduce the radii to satisfy constraint.
                # This is a local fix.
                excess = sum_r - dist
                # Reduce both radii by excess/2
                radii[i] -= excess / 2
                radii[j] -= excess / 2
                # Update dists? No, centers fixed.
                # But reducing radii might cause new overlaps with others?
                # Unlikely if excess is small.
    
    # Re-check validity
    # If still invalid, we might need to re-run or accept lower sum.
    # But usually this logic works for small overlaps.
    
    # Compute final sum
    final_sum = np.sum(radii)
    
    return centers, radii, final_sum

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation
    # (Assuming validate_packing is available or we check manually)
    # For local testing

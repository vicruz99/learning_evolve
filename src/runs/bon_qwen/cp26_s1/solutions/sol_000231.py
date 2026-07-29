# sol_000231 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dc099519) state=56a3bc98 sum of radii=1.491395 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.RandomState(42)

    # 1. Initialization
    # Create a 5x5 grid and add 1 point
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    grid_pts = np.array([[x, y] for x in grid_x for y in grid_y])
    
    # Add 26th point in a gap, e.g., (0.2, 0.2) is a gap center in 5x5 grid with spacing 0.2
    # Grid points: 0.1, 0.3, 0.5, 0.7, 0.9
    # Gap at 0.2 is between 0.1 and 0.3.
    # Distance to (0.1, 0.1) is sqrt(0.1^2 + 0.1^2) ~ 0.141.
    extra_pt = np.array([0.2, 0.2])
    
    centers = np.vstack([grid_pts, extra_pt])
    
    # Shuffle to break symmetry
    centers = centers[rng.permutation(n)]
    
    # Initial radii
    radii = np.full(n, 0.05)

    # 2. Force-directed relaxation
    # Indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    dt = 0.005
    r_growth = 0.0001
    
    for step in range(3000):
        # Compute distances and overlaps
        diff = centers[i_idx] - centers[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        r_sums = radii[i_idx] + radii[j_idx]
        
        overlaps = r_sums - dists
        
        # Repulsive forces
        forces = np.zeros_like(centers)
        
        # Only process overlapping pairs to save time, but for n=26 all is fast
        # We want to push apart if dist < r_sum
        # Force direction: diff / dist
        # Magnitude: overlap
        
        # Safe dist
        safe_dists = np.maximum(dists, 1e-9)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        
        # Spring force
        f_mag = overlaps * 5.0
        # Add to forces
        np.add.at(forces, i_idx, unit_vecs * f_mag[:, np.newaxis])
        np.add.at(forces, j_idx, -unit_vecs * f_mag[:, np.newaxis])
        
        # Boundary forces
        # Keep inside [r, 1-r]
        # If x < r, push right. If x > 1-r, push left.
        # Violation amount
        v_x = np.maximum(radii - centers[:, 0], 0) - np.maximum(centers[:, 0] + radii - 1.0, 0)
        v_y = np.maximum(radii - centers[:, 1], 0) - np.maximum(centers[:, 1] + radii - 1.0, 0)
        
        forces[:, 0] += v_x * 5.0
        forces[:, 1] += v_y * 5.0
        
        # Update centers
        # Add noise that decreases over time (simulated annealing)
        noise = rng.normal(0, 0.002 * np.exp(-step/500), size=centers.shape)
        centers += forces * dt + noise
        
        # Clip centers to [0, 1] just in case, though forces should keep them in [r, 1-r]
        # But if r is small, [0,1] is safe.
        centers[:, 0] = np.clip(centers[:, 0], 0.0, 1.0)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, 1.0)
        
        # Grow radii
        radii += r_growth
        
        # Slow down growth
        if step > 1500:
            r_growth *= 0.995

    # 3. Refine with optimization
    # We want to maximize min_separation / 2 subject to boundary constraints.
    # This is equivalent to finding max r such that 26 circles of radius r fit.
    # But we have fixed centers now.
    # We can optimize centers to maximize the "clearance".
    
    # Objective: maximize min(d_ij/2 - r_i, min_boundary - r_i) ?
    # Actually, if we assume equal radii r, we want to maximize r.
    # r <= d_ij / 2 for all i,j
    # r <= min_boundary for all i
    # So r = min( min_{i,j} d_ij / 2, min_i min_boundary_i )
    # We want to maximize this r by moving centers.
    
    # Let's define a function that returns the max feasible equal radius for a given set of centers.
    # Then maximize this radius.
    
    def objective(vars):
        c = vars.reshape(n, 2)
        # Compute pairwise distances
        # Vectorized distance matrix
        # c[:, None, :] - c[None, :, :]
        # shape (n, n, 2)
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diffs, axis=2)
        # Lower triangle (unique pairs)
        np.fill_diagonal(dists_mat, np.inf)
        min_d = np.min(dists_mat)
        
        # Boundary distances
        # x, 1-x, y, 1-y
        bounds = np.column_stack([c[:, 0], 1.0 - c[:, 0], c[:, 1], 1.0 - c[:, 1]])
        min_bound = np.min(bounds)
        
        # Max radius is min(min_d/2, min_bound)
        r_max = min(min_d / 2.0, min_bound)
        
        # We want to maximize r_max, so minimize negative
        return -r_max

    # Use scipy minimize
    # Bounds for centers: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n)
    
    res = minimize(objective, centers.flatten(), method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    optimal_centers = res.x.reshape(n, 2)
    optimal_r = -res.fun
    
    # Set all radii to optimal_r
    optimal_radii = np.full(n, optimal_r)
    
    # Check if this solution is valid (it should be by construction)
    # And compute sum
    sum_radii = np.sum(optimal_radii)
    
    # Wait, is equal radii optimal?
    # For n=26, maybe variable radii allow better sum.
    # But equal radii is a very strong candidate.
    # Let's see if we can tweak variable radii.
    # If we have a valid packing with radius r, sum = 26r.
    # If we can increase some radii and decrease others, sum might change.
    # But as discussed, equal is likely best.
    
    # Let's return the result.
    # However, the optimizer might have found a local minimum.
    # The force simulation helped escape local minima.
    
    # One detail: The Nelder-Mead might be slow or get stuck.
    # But with 52 variables, it's manageable.
    
    # Let's also ensure the centers are within [0,1] (optimizer bounds handle this? Nelder-Mead doesn't support bounds well).
    # We should use a method that respects bounds or clip.
    # But the objective penalizes being outside (min_bound becomes negative? No, min_bound can be negative if center is outside).
    # If center is at -0.1, 1-x = 1.1, x = -0.1. min is -0.1.
    # r_max becomes negative?
    # We should penalize negative r_max or just restrict search space.
    # Since we initialized inside, and objective pushes to increase r, it should stay inside?
    # Not necessarily.
    # Let's use 'L-BFGS-B' with bounds if we modify objective to be smooth?
    # Objective has min() which is non-smooth.
    # Nelder-Mead handles non-smooth but not bounds.
    # We can just clip centers in the objective?
    # No, that changes topology.
    # Let's just hope Nelder-Mead stays inside.
    # Or use a penalty in objective for being outside.
    
    def objective_with_penalty(vars):
        c = vars.reshape(n, 2)
        # Penalty for outside
        penalty = 0
        if np.any(c < 0) or np.any(c > 1):
             # Heuristic penalty
             penalty = 1000 * (np.sum(np.maximum(0, -c)) + np.sum(np.maximum(0, c - 1)))
        
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists_mat, np.inf)
        min_d = np.min(dists_mat)
        
        bounds = np.column_stack([c[:, 0], 1.0 - c[:, 0], c[:, 1], 1.0 - c[:, 1]])
        min_bound = np.min(bounds)
        
        r_max = min(min_d / 2.0, min_bound)
        
        return -r_max + penalty

    # Re-run optimization with penalty
    res2 = minimize(objective_with_penalty, centers.flatten(), method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-7})
    
    final_centers = res2.x.reshape(n, 2)
    final_r = -res2.fun
    
    # If final_r is negative (due to penalty), clamp to 0?
    # But it should be positive.
    # Let's re-evaluate r without penalty to be sure.
    
    # Recalculate r for final_centers
    diffs = final_centers[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
    dists_mat = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists_mat, np.inf)
    min_d = np.min(dists_mat)
    bounds = np.column_stack([final_centers[:, 0], 1.0 - final_centers[:, 0], final_centers[:, 1], 1.0 - final_centers[:, 1]])
    min_bound = np.min(bounds)
    true_r = min(min_d / 2.0, min_bound)
    
    if true_r < 0:
        # Fallback to a valid packing, e.g., small circles
        true_r = 0.01
        final_centers = np.random.rand(n, 2) # Invalid, but r small will fix?
        # Actually just grid
        final_centers = np.vstack([grid_pts, extra_pt])
        true_r = 0.05 # Safe

    final_radii = np.full(n, true_r)
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3ad176de) state=b2afbb56 sum of radii=2.075993 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Seed for reproducibility and multiple restarts
    rng = np.random.RandomState(42)
    
    # Helper to initialize centers in a grid-like pattern
    def get_initial_centers():
        # 5x5 grid plus one, slightly perturbed
        centers = np.zeros((n, 2))
        # 5 rows, 5 cols for 25
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                centers[idx] = [0.1 + j * 0.2, 0.1 + i * 0.2]
        # 26th circle in the center or a gap
        centers[25] = [0.5, 0.5]
        
        # Add random jitter
        centers += rng.uniform(-0.02, 0.02, size=centers.shape)
        # Clamp to valid range (0.05 to 0.95)
        centers = np.clip(centers, 0.05, 0.95)
        return centers

    for restart in range(5):
        centers = get_initial_centers()
        # Add extra unique jitter for each restart
        centers += rng.uniform(-0.05, 0.05, size=centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Optimization Loop
        for iteration in range(100):
            # 1. Compute distances and boundary limits
            # Pairwise distances
            dists = np.sqrt(np.sum((centers[:, np.newaxis] - centers[np.newaxis, :]) ** 2, axis=2))
            
            # Boundary limits for each circle
            b_x = np.minimum(centers[:, 0], 1 - centers[:, 0])
            b_y = np.minimum(centers[:, 1], 1 - centers[:, 1])
            b_i = np.minimum(b_x, b_y)
            
            # 2. Setup and Solve LP for radii
            # Maximize sum(r_i) <=> Minimize -sum(r_i)
            c_obj = -np.ones(n)
            
            # Constraints: r_i + r_j <= dist_ij
            # We only need upper triangle to avoid redundancy
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            n_pairs = len(pairs)
            
            # A_ub matrix for LP
            # Rows for pairs
            A_ub_pairs = np.zeros((n_pairs, n))
            b_ub_pairs = np.zeros(n_pairs)
            for k, (i, j) in enumerate(pairs):
                A_ub_pairs[k, i] = 1
                A_ub_pairs[k, j] = 1
                b_ub_pairs[k] = dists[i, j]
            
            # Combine with boundary constraints (if we want duals for them, we include in A_ub)
            # But linprog handles bounds easily. However, for force calculation on centers,
            # boundary forces are simple (push away from wall).
            
            A_ub = A_ub_pairs
            b_ub = b_ub_pairs
            
            # Bounds for radii: 0 <= r_i <= b_i
            bounds = [(0, b_i[k]) for k in range(n)]
            
            # Solve LP
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if not res.success:
                continue
                
            radii = res.x
            current_sum = np.sum(radii)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_radii = radii.copy()
                best_centers = centers.copy()
            
            # 3. Compute forces based on duals to update centers
            # Duals for pairwise constraints
            duals = res.ineqlin.marginals
            
            forces = np.zeros_like(centers)
            
            for k, (i, j) in enumerate(pairs):
                lam = duals[k]
                if lam > 1e-6: # Only active constraints contribute
                    d = dists[i, j]
                    if d > 1e-9:
                        dir_vec = (centers[i] - centers[j]) / d
                        force_mag = lam # Gradient of distance is unit vector
                        forces[i] += force_mag * dir_vec
                        forces[j] -= force_mag * dir_vec
            
            # Add boundary forces (push away from walls if limited)
            # If radius is close to boundary limit, push center towards center of square
            # This is a heuristic to help centers move into more open space
            margin = 0.0
            for i in range(n):
                if radii[i] > 0:
                    # Check X
                    if centers[i, 0] - radii[i] < margin:
                        forces[i, 0] += 0.5 # Push right
                    if centers[i, 0] + radii[i] > 1 - margin:
                        forces[i, 0] -= 0.5 # Push left
                    # Check Y
                    if centers[i, 1] - radii[i] < margin:
                        forces[i, 1] += 0.5 # Push up
                    if centers[i, 1] + radii[i] > 1 - margin:
                        forces[i, 1] -= 0.5 # Push down

            # 4. Update centers
            step_size = 0.005
            centers += step_size * forces
            
            # Clamp centers to valid range [0, 1]
            centers = np.clip(centers, 1e-6, 1 - 1e-6)
            
            # Check for convergence (optional, but loop count is fixed)
            if iteration > 50 and abs(np.sum(radii) - best_sum) < 1e-8:
                # If radii didn't improve in best tracking, maybe we are stuck?
                # Actually best_sum tracks history, let's check current vs previous radii sum
                pass 
                
            # Check overlap validity (safety)
            # If LP was solved correctly, it should be valid, but numerical errors might occur.
            # We can re-scale radii slightly if needed, but LP guarantees r_i + r_j <= dist.
            
    # Final validation and formatting
    # Ensure radii and centers are valid
    # Sometimes LP might give 0 radius or slight violations due to float precision.
    # We clamp radii to be non-negative.
    best_radii = np.maximum(best_radii, 0)
    
    # Re-validate using the provided logic implicitly
    # Check if any circles are outside
    valid = True
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            # Fix minor boundary violations by shrinking
            if x - r < 0: r = x
            if x + r > 1: r = 1 - x
            if y - r < 0: r = min(r, y)
            if y + r > 1: r = min(r, 1 - y)
            best_radii[i] = max(r, 0)
            
    # Final sum
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum

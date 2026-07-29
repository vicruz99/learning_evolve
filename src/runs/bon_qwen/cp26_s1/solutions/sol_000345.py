# sol_000345 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2c580e0d) state=d5563a17 sum of radii=2.386120 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a local search optimization on a hexagonal lattice initialization.
    """
    N = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Helper to compute radii for a given set of centers
    def get_radii(centers):
        r = np.zeros(N)
        # Distance to boundaries
        dist_bdry = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                               np.minimum(centers[:, 1], 1 - centers[:, 1]))
        
        # Distance to other centers
        # Compute pairwise distances
        # centers shape (N, 2)
        # dists shape (N, N)
        # Use broadcasting
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Radius is min of (dist to boundary, min(dist to neighbor)/2)
        # Diagonal of dists is 0, ignore
        np.fill_diagonal(dists, np.inf)
        min_neighbor_dist = np.min(dists, axis=1)
        
        r = np.minimum(dist_bdry, 0.5 * min_neighbor_dist)
        return r

    # Objective function
    def objective(centers):
        r = get_radii(centers)
        return np.sum(r)

    # Generate initial configuration
    # Try a few different initializations and pick the best
    def generate_initial_config(seed=0):
        rng = np.random.default_rng(seed)
        centers = np.zeros((N, 2))
        
        # Hexagonal packing approximation
        # Estimate radius for 26 circles to determine spacing
        # Area approx: 26 * pi * r^2 ~ 0.9 * 1 => r ~ 0.105
        # Spacing 2r ~ 0.21
        # Vertical spacing sqrt(3)/2 * 2r ~ 0.18
        
        # Let's try to fit rows
        # 6 rows seems appropriate
        # Row 0, 2, 4: 5 circles? Row 1, 3, 5: 4 circles?
        # 5+4+5+4+5+4 = 27. Too many.
        # 5+4+5+4+5+3 = 26.
        
        # Let's just generate a dense grid and pick 26, or random
        # A slightly perturbed hex lattice is good.
        
        r_est = 0.1
        h = math.sqrt(3) * r_est
        w = 2 * r_est
        
        # Fill a grid
        points = []
        y = r_est
        row = 0
        while y <= 1 - r_est:
            shift = r_est if row % 2 == 1 else 0
            x = r_est + shift
            while x <= 1 - r_est:
                points.append([x, y])
                x += w
            y += h
            row += 1
            
        if len(points) >= N:
            # If we have too many, maybe pick first N or random N?
            # Actually, we need exactly 26. 
            # If grid is too sparse, this loop might not fill.
            # Let's use random init if grid fails or is too small.
            pass
        
        if len(points) < N:
            # Fallback to random initialization
            centers = rng.uniform(0.1, 0.9, size=(N, 2))
        else:
            # Use the first N points from the lattice? 
            # Or distribute them better.
            # Let's just use random to be safe and robust for optimization
            centers = rng.uniform(0.05, 0.95, size=(N, 2))
            
        return centers

    # Optimization loop (Simulated Annealing / Hill Climbing hybrid)
    def optimize(init_centers, iterations=20000, temp_init=0.05, temp_final=1e-5):
        centers = init_centers.copy()
        current_val = objective(centers)
        best_val = current_val
        best_centers = centers.copy()
        
        # Initial radius estimate for step size
        r_avg = np.mean(get_radii(centers))
        step_size = r_avg * 0.5
        
        for i in range(iterations):
            # Cooling schedule
            temp = temp_init * (temp_final / temp_init) ** (i / iterations)
            
            # Perturb a random circle
            idx = np.random.randint(0, N)
            new_centers = centers.copy()
            
            # Move the chosen circle
            # Step size adapts to temperature or is fixed?
            # Fixed step size relative to radii is usually better
            delta_x = np.random.uniform(-step_size, step_size)
            delta_y = np.random.uniform(-step_size, step_size)
            
            new_centers[idx, 0] += delta_x
            new_centers[idx, 1] += delta_y
            
            # Clip to valid range (inside square with margin 0)
            # Actually centers can be anywhere, radii handle boundaries.
            # But keeping centers inside [0,1] is logical.
            # If center is outside, radius becomes 0 or negative?
            # Our get_radii handles boundary distance. 
            # If center is at -0.1, dist to boundary is negative? 
            # np.minimum handles it, but physically radius can't be negative.
            # Let's clip centers to [0, 1] to be safe, or let radii be 0.
            # Actually, if center is outside, dist to boundary is negative, 
            # so radius becomes negative? No, dist_bdry is min(x, 1-x...).
            # If x < 0, dist is x (negative). Radius becomes negative.
            # Validation requires radii >= 0.
            # So we must ensure centers are in [0,1] or handle negative radii.
            # Let's just clip centers to [epsilon, 1-epsilon]?
            # No, optimization might need to push boundaries.
            # But radius is 0 if outside.
            # Let's constrain moves to keep centers in [0,1].
            
            # Apply move
            # Check bounds
            if not (0 <= new_centers[idx, 0] <= 1 and 0 <= new_centers[idx, 1] <= 1):
                # Reflect or reject?
                # Just reject move if out of bounds for simplicity
                continue
            
            new_val = objective(new_centers)
            diff = new_val - current_val
            
            # Acceptance criteria
            if diff > 0 or np.random.random() < math.exp(diff / max(temp, 1e-9)):
                centers = new_centers
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_centers = centers.copy()
            
            # Adaptive step size?
            # If many rejections, decrease step?
            # For simplicity, keep fixed or slowly decay.
            
        return best_centers, best_val

    # Run multiple restarts
    best_overall_centers = None
    best_overall_sum = 0.0
    
    # We can try a few seeds
    for seed in range(5):
        # Create a hexagonal-ish start for better convergence
        centers_init = np.zeros((N, 2))
        # Generate a hex grid
        r_guess = 0.1
        h = math.sqrt(3) * r_guess
        w = 2 * r_guess
        
        points = []
        y = r_guess
        row_idx = 0
        while y <= 1 - r_guess + 1e-9:
            shift = r_guess if row_idx % 2 == 1 else 0
            x = r_guess + shift
            while x <= 1 - r_guess + 1e-9:
                points.append([x, y])
                x += w
            y += h
            row_idx += 1
        
        # If we have enough points, use them, else random
        rng = np.random.default_rng(seed)
        if len(points) >= N:
            # Shuffle and take N
            pts_arr = np.array(points)
            indices = rng.choice(len(pts_arr), N, replace=False)
            centers_init = pts_arr[indices]
            # Add small noise
            centers_init += rng.uniform(-0.01, 0.01, size=centers_init.shape)
            centers_init = np.clip(centers_init, 1e-4, 1 - 1e-4)
        else:
            centers_init = rng.uniform(0.1, 0.9, size=(N, 2))
            
        opt_centers, opt_sum = optimize(centers_init, iterations=5000, temp_init=0.1, temp_final=1e-6)
        
        if opt_sum > best_overall_sum:
            best_overall_sum = opt_sum
            best_overall_centers = opt_centers

    # Final refinement with fixed centers calculation
    final_radii = get_radii(best_overall_centers)
    
    # Ensure no negative radii due to precision
    final_radii = np.maximum(final_radii, 0.0)
    
    return best_overall_centers, final_radii, np.sum(final_radii)

# For testing locally
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")

# sol_000187 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9afef83a) state=b4cfc0bd sum of radii=1.339996 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import scipy.spatial

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

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal lattice initialization followed by boundary repulsion 
    and radius maximization.
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # We perform multiple runs with different initial seeds to escape local minima
    num_runs = 30
    
    for run in range(num_runs):
        # 1. Initialize centers in a hexagonal grid pattern
        # This is a heuristic to place circles efficiently
        centers = np.zeros((n, 2))
        idx = 0
        y_pos = 0.0
        x_start = 0.0
        
        # Simple hexagonal packing generator
        while idx < n and y_pos < 1.2:
            row_circles = 0
            while x_start + row_circles * 1.0 < 1.2 and idx < n:
                centers[idx, 0] = x_start + row_circles
                centers[idx, 1] = y_pos
                idx += 1
                row_circles += 1
            x_start += 0.5
            y_pos += np.sqrt(3) / 2

        # If we didn't fill all (due to rough grid logic), fill remaining randomly or skip
        if idx < n:
            # Fill remaining with random positions inside the square
            for k in range(idx, n):
                centers[k, 0] = np.random.uniform(0.2, 0.8)
                centers[k, 1] = np.random.uniform(0.2, 0.8)

        # 2. Add random noise to break symmetry and help optimization
        centers += np.random.uniform(-0.05, 0.05, centers.shape)
        
        # Clip to valid range for the optimizer to start
        centers = np.clip(centers, 0.01, 0.99)

        # 3. Optimize positions to maximize the minimum distance between circles and boundaries
        # This is equivalent to maximizing the radius of equal circles.
        # We define a "safety radius" for each point based on constraints.
        
        def objective(x):
            # x is the flattened centers array
            curr_centers = x.reshape((n, 2))
            
            # Penalty for being too close to boundaries
            # We want to maximize distance to walls
            dists_to_walls = np.minimum(curr_centers, 1.0 - curr_centers)
            min_wall_dist = np.min(dists_to_walls)
            
            # Penalty for being too close to other circles
            # We want to maximize the distance between any pair
            diffs = curr_centers[:, np.newaxis, :] - curr_centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            
            # Set diagonal to infinity so we don't consider self-distance
            dists[np.eye(n, dtype=bool)] = np.inf
            min_pair_dist = np.min(dists)
            
            # We want to maximize both min_wall_dist and min_pair_dist / 2
            # A simple combined objective is to maximize the minimum of these
            # r_effective = min(min_wall_dist, min_pair_dist / 2)
            # We maximize r_effective
            return -np.minimum(min_wall_dist, min_pair_dist / 2)

        try:
            # Use Nelder-Mead or L-BFGS-B. Nelder-Mead is robust for non-smooth min functions
            result = scipy.optimize.minimize(
                objective, 
                centers.flatten(), 
                method='Nelder-Mead', 
                options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 2000}
            )
            
            optimized_centers = result.x.reshape((n, 2))
            
            # 4. Calculate maximum radii for each circle based on optimized positions
            radii = np.zeros(n)
            for i in range(n):
                # Distance to boundaries
                x, y = optimized_centers[i]
                r_max_boundary = min(x, 1-x, y, 1-y)
                
                # Distance to other centers minus their potential radii?
                # Since we optimized for equal radii implicitly, we can just take half distance to nearest neighbor
                # However, to be precise for unequal radii, we calculate the max radius such that 
                # it doesn't overlap with neighbors.
                # A simple valid radius is half the distance to the nearest neighbor, 
                # capped by boundary distance.
                
                min_dist_to_other = np.inf
                for j in range(n):
                    if i != j:
                        dist = np.sqrt(np.sum((optimized_centers[i] - optimized_centers[j])**2))
                        if dist < min_dist_to_other:
                            min_dist_to_other = dist
                
                # Radius is limited by distance to boundary and distance to nearest neighbor
                # r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
                # r_i + r_j <= dist_ij. Assuming r_j is also large, a safe lower bound for r_i is dist_ij / 2
                # But we want to maximize sum. The tightest constraint for r_i is usually the nearest neighbor.
                # If we set all radii to be determined by the "bottleneck" of the packing:
                # r_i = min(r_max_boundary, min_{j!=i} (dist_ij / 2))
                
                radii[i] = min(r_max_boundary, min_dist_to_other / 2)

            current_sum = np.sum(radii)
            
            # Validate just to be sure
            if validate_packing(optimized_centers, radii):
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = optimized_centers.copy()
                    best_radii = radii.copy()

        except Exception as e:
            continue

    return best_centers, best_radii, best_sum

# To execute the packing
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")

# sol_000022 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d7881bb) state=8caa9ba1 sum of radii=2.044252 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import random
import math

def dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def get_radii(centers):
    """
    Computes a valid set of radii for given centers.
    r_i = 0.5 * min(dist to boundary, min dist to other centers)
    This ensures r_i + r_j <= dist(i, j) and circles are inside square.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Distance to boundaries
    dists_to_boundary = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1])
    )
    
    # Distance to nearest neighbor
    dists_to_nn = np.full(n, np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(centers[i], centers[j])
            if d < dists_to_nn[i]:
                dists_to_nn[i] = d
            if d < dists_to_nn[j]:
                dists_to_nn[j] = d
                
    # Radius is half the minimum of boundary distance and neighbor distance
    # Actually, r_i <= dist_to_boundary AND r_i <= dist_to_nn / 2 is not strictly required by r_i + r_j <= d.
    # The condition r_i <= d_ij / 2 for all j is sufficient for r_i + r_j <= d_ij?
    # If r_i = d_ij/2 and r_j = d_ij/2, sum is d_ij. OK.
    # But r_i could be larger if r_j is smaller.
    # However, the heuristic r_i = 0.5 * min(dist_to_boundary, min_j d_ij) is safe and simple.
    # Let's use the safe heuristic.
    
    for i in range(n):
        r = 0.5 * min(dists_to_boundary[i], dists_to_nn[i])
        radii[i] = r
        
    return radii

def score(centers):
    radii = get_radii(centers)
    return np.sum(radii)

def run_packing():
    n = 26
    
    # 1. Initialize with a Hexagonal-like grid
    # We want to pack 26 points. 
    # A 5x5 grid is 25. Let's try to generate a hexagonal lattice and pick 26 points.
    # Or just a perturbed grid.
    
    # Generate a set of candidate points on a triangular lattice
    # Spacing 1.0 initially
    pts = []
    # Row 0
    for i in range(6):
        pts.append([i, 0])
    # Row 1
    for i in range(5):
        pts.append([i + 0.5, math.sqrt(3)/2])
    # Row 2
    for i in range(6):
        pts.append([i, math.sqrt(3)])
    # Row 3
    for i in range(5):
        pts.append([i + 0.5, 1.5 * math.sqrt(3)])
    # Row 4
    for i in range(4): # 4 points to make total 26? 6+5+6+5+4 = 26.
        pts.append([i, 2 * math.sqrt(3)])
        
    pts = np.array(pts)
    
    # Center and scale to fit in [0,1]x[0,1] roughly
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale factor to fit with some margin
    # We want to fit inside, so scale by 1 / max(width, height) roughly, but keep aspect ratio?
    # Actually, we can scale x and y independently? No, lattice structure should be preserved?
    # Maybe just uniform scaling.
    scale = 0.95 / max(width, height) # 0.95 to leave margin for centers
    
    # Center in square
    centers = (pts - np.array([min_x, min_y])) * scale
    # Shift to center
    centers = centers - centers.mean(axis=0) + 0.5
    
    # Clip to stay strictly inside initially to avoid issues, though optimization should handle it
    # But get_radii handles boundaries.
    
    # 2. Local Search Optimization
    # We will perturb centers to maximize sum of radii.
    
    best_centers = centers.copy()
    best_score = score(best_centers)
    
    # Simulated Annealing / Random Hill Climbing
    temp = 0.1 # Initial perturbation step size
    step = 0.02
    
    # Run for a number of iterations
    # Since n is small, we can run many iterations
    num_iterations = 5000
    
    for _ in range(num_iterations):
        # Pick a random circle
        idx = random.randint(0, n - 1)
        
        # Generate a random move
        dx = random.uniform(-step, step)
        dy = random.uniform(-step, step)
        
        new_centers = best_centers.copy()
        new_centers[idx] += np.array([dx, dy])
        
        # Keep inside [0,1]
        new_centers[idx] = np.clip(new_centers[idx], 0, 1)
        
        # To avoid circles sticking to boundaries which reduces radius, maybe bias away?
        # But get_radii handles it.
        
        current_score = score(new_centers)
        
        # Accept if better
        if current_score > best_score:
            best_centers = new_centers
            best_score = current_score
            # Occasionally reduce step size?
            if random.random() < 0.001:
                step *= 0.95
        else:
            # Accept with some probability (Simulated Annealing)
            # Probability exp((new - old)/temp)
            # But score difference can be small.
            # Let's just do greedy mostly, but allow small worsening to escape local optima?
            # Or just restart if stuck?
            pass
            
        # Decrease step size slowly
        step *= 0.9999

    # 3. Refine: Coordinate descent or more aggressive perturbation
    # Try to jiggle each circle individually
    for _ in range(100):
        improved = False
        for i in range(n):
            # Try moving circle i in random directions
            best_local_centers = best_centers.copy()
            best_local_score = best_score
            
            # Try 4 directions + stay
            directions = [
                np.array([0.01, 0]), np.array([-0.01, 0]),
                np.array([0, 0.01]), np.array([0, -0.01]),
                np.array([0, 0])
            ]
            
            for d in directions:
                temp_centers = best_centers.copy()
                temp_centers[i] += d
                temp_centers[i] = np.clip(temp_centers[i], 0, 1)
                s = score(temp_centers)
                if s > best_local_score:
                    best_local_centers = temp_centers
                    best_local_score = s
                    improved = True
            
            best_centers = best_local_centers
            best_score = best_local_score
        
        if not improved:
            break

    # Final Radii Calculation
    radii = get_radii(best_centers)
    sum_radii = np.sum(radii)
    
    # Validation check (internal)
    # validate_packing is read-only, but we can check logic
    # Ensure no overlaps with computed radii
    # Our heuristic ensures r_i + r_j <= d_ij / 2 + d_ij / 2 = d_ij ?
    # Wait, r_i = 0.5 * min(..., min_k d_ik).
    # So r_i <= 0.5 * d_ij for all j.
    # So r_i + r_j <= 0.5 d_ij + 0.5 d_ij = d_ij.
    # Yes, valid.
    
    return best_centers, radii, sum_radii

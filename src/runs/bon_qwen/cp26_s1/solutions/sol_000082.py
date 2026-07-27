# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9821b492) state=bc97a7c5 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def is_valid_packing(centers, radii):
    """
    Checks if the given centers and radii form a valid packing within the unit square.
    """
    n = centers.shape[0]
    # Check boundaries
    # Circle i is inside if x_i - r_i >= 0, x_i + r_i <= 1, etc.
    if np.any(centers[:, 0] < radii - 1e-12): return False
    if np.any(centers[:, 0] > 1 - radii + 1e-12): return False
    if np.any(centers[:, 1] < radii - 1e-12): return False
    if np.any(centers[:, 1] > 1 - radii + 1e-12): return False
    
    # Check overlaps
    # Distance between centers must be >= sum of radii
    # Compute pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    if np.any(dists < r_sum - 1e-12):
        return False
    return True

def relax_centers(centers, radii, max_steps=20):
    """
    Resolves overlaps and boundary violations by moving centers.
    Returns the updated centers.
    """
    n_c = len(centers)
    for _ in range(max_steps):
        moved = False
        
        # 1. Boundary correction
        # Push centers back if they are too close to walls
        # Left wall
        mask = centers[:, 0] < radii
        if np.any(mask):
            centers[mask, 0] = radii[mask]
            moved = True
        # Right wall
        mask = centers[:, 0] > 1 - radii
        if np.any(mask):
            centers[mask, 0] = 1 - radii[mask]
            moved = True
        # Bottom wall
        mask = centers[:, 1] < radii
        if np.any(mask):
            centers[mask, 1] = radii[mask]
            moved = True
        # Top wall
        mask = centers[:, 1] > 1 - radii
        if np.any(mask):
            centers[mask, 1] = 1 - radii[mask]
            moved = True

        # 2. Pairwise push to resolve overlaps
        # Compute differences and distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = r_sum - dists
        
        # Iterate over unique pairs
        for i in range(n_c):
            for j in range(i + 1, n_c):
                ov = overlap[i, j]
                if ov > 1e-12:
                    d = dists[i, j]
                    if d < 1e-9:
                        # Centers coincide, push randomly
                        dx, dy = 1e-5, 0
                    else:
                        dx = diffs[i, j, 0] / d
                        dy = diffs[i, j, 1] / d
                    
                    # Push apart by half the overlap each
                    push = ov / 2.0
                    centers[i, 0] -= dx * push
                    centers[i, 1] -= dy * push
                    centers[j, 0] += dx * push
                    centers[j, 1] += dy * push
                    moved = True
        
        if not moved:
            break
    return centers

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Patterns for hexagonal grid initialization
    # Row counts summing to 26
    patterns = [
        [5, 5, 5, 5, 3, 3],
        [5, 5, 5, 4, 4, 3],
        [5, 5, 4, 5, 4, 3],
        [5, 4, 5, 4, 5, 3],
        [6, 5, 5, 5, 5]
    ]
    
    # Starting radius for grid layout
    r_start = 0.09 
    
    for row_counts in patterns:
        if sum(row_counts) != n:
            continue
            
        centers = []
        radii = []
        
        # Vertical spacing for hexagonal packing
        h_step = r_start * math.sqrt(3)
        y_pos = r_start
        
        for r_idx, count in enumerate(row_counts):
            # Shift alternate rows to create hexagonal lattice
            # Even rows (0-indexed) start at x = r
            # Odd rows start at x = 2r (shifted by r)
            
            if r_idx % 2 == 0:
                slots = []
                k = 0
                while True:
                    x = r_start + k * 2 * r_start
                    if x + r_start <= 1.0 + 1e-9:
                        slots.append(x)
                    else:
                        break
                    k += 1
            else:
                slots = []
                k = 0
                while True:
                    x = 2 * r_start + k * 2 * r_start
                    if x + r_start <= 1.0 + 1e-9:
                        slots.append(x)
                    else:
                        break
                    k += 1
            
            # If not enough slots in pattern, distribute evenly
            if len(slots) < count:
                slots = np.linspace(r_start, 1 - r_start, count).tolist()
            
            # Pick required number of slots
            chosen = slots[:count]
            for x in chosen:
                centers.append([x, y_pos])
                radii.append(r_start)
            
            y_pos += h_step
        
        if len(centers) == n:
            centers = np.array(centers)
            radii = np.array(radii)
            
            # Initial relaxation to fix any layout issues
            centers = relax_centers(centers, radii, max_steps=50)
            
            # Check initial validity
            if is_valid_packing(centers, radii):
                s = np.sum(radii)
                if s > best_sum_radii:
                    best_sum_radii = s
                    best_centers = centers.copy()
                    best_radii = radii.copy()
            
            # Expansion loop
            for step in range(2000):
                # Relax to fix overlaps caused by previous expansion or initial state
                centers = relax_centers(centers, radii, max_steps=10)
                
                if is_valid_packing(centers, radii):
                    s = np.sum(radii)
                    if s > best_sum_radii:
                        best_sum_radii = s
                        best_centers = centers.copy()
                        best_radii = radii.copy()
                    # Try to expand
                    radii += 0.0002
                else:
                    # If invalid, shrink slightly to recover valid state
                    radii *= 0.999

    # Random restarts to find better local optima
    np.random.seed(42)
    for _ in range(5):
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.05)
        
        centers = relax_centers(centers, radii, max_steps=100)
        if is_valid_packing(centers, radii):
            s = np.sum(radii)
            if s > best_sum_radii:
                best_sum_radii = s
                best_centers = centers.copy()
                best_radii = radii.copy()
        
        for step in range(2000):
            centers = relax_centers(centers, radii, max_steps=10)
            if is_valid_packing(centers, radii):
                s = np.sum(radii)
                if s > best_sum_radii:
                    best_sum_radii = s
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                radii += 0.0002
            else:
                radii *= 0.999

    # Fallback if no valid packing found
    if best_centers is None:
        best_centers = np.random.rand(n, 2)
        best_radii = np.zeros(n)
        best_sum_radii = 0.0

    return best_centers, best_radii, best_sum_radii

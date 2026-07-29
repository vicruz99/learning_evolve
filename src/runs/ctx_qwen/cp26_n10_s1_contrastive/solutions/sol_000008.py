# sol_000008 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 92133c71) state=1d17a383 sum of radii=2.075329 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def get_initial_hexagonal_centers(n, width=1.0, height=1.0):
    """
    Generate initial centers for n circles in a hexagonal pattern.
    """
    centers = []
    # Try to fit rows
    # Estimate radius to start with, e.g., 0.08
    r_est = 0.08
    
    y = r_est
    row = 0
    while len(centers) < n:
        x = r_est
        while x + r_est <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_est
        row += 1
        y += math.sqrt(3) * r_est
        # Shift every other row
        if row % 2 == 1:
            offset = r_est
            x = r_est + offset
            while x + r_est <= 1.0 and len(centers) < n:
                # Check if we already added this spot in previous logic? 
                # The logic above adds full rows. Let's restart logic for proper hex.
                pass 
        
    # Better hex generation
    centers = []
    r_est = 0.08
    y = r_est
    shift = 0
    while len(centers) < n:
        x = r_est + shift
        while x + r_est <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_est
        y += math.sqrt(3) * r_est
        shift = r_est if shift == 0 else 0
        
    return np.array(centers)

def solve_radii_lp(centers):
    """
    Given fixed centers, solve LP to maximize sum of radii.
    Maximize sum(r_i)
    Subject to:
      r_i + r_j <= dist(i, j) for all i < j
      r_i <= x_i
      r_i <= 1 - x_i
      r_i <= y_i
      r_i <= 1 - y_i
      r_i >= 0
    """
    n = centers.shape[0]
    
    # Objective: minimize -sum(r) => c = -1 for all r
    c = np.ones(n) * -1.0
    
    # Inequality constraints A_ub @ r <= b_ub
    # r_i + r_j <= dist
    constraints_A = []
    constraints_b = []
    
    # Precompute distances
    # dist matrix
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Pairwise constraints
    # There are n*(n-1)/2 constraints. For n=26, ~325. Manageable.
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    # This can be handled by upper bounds in linprog, but linprog supports bounds per variable.
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        # Clip to non-negative
        max_r = max(0.0, max_r)
        bounds.append((0.0, max_r))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, res.fun * -1.0
        else:
            return None, -np.inf
    except Exception:
        return None, -np.inf

def perturb_centers(centers, radii, step_size=1e-4, iterations=50):
    """
    Perturb centers to increase the potential sum of radii.
    Simple heuristic: move centers away from neighbors that constrain radii.
    """
    n = centers.shape[0]
    best_centers = centers.copy()
    best_sum = 0
    
    # First, solve LP to get baseline
    current_radii, current_sum = solve_radii_lp(centers)
    if current_radii is not None:
        best_sum = current_sum
        best_radii = current_radii
    else:
        # Fallback
        current_radii = np.ones(n) * 0.01
        best_radii = current_radii

    # Force-based relaxation
    # If r_i + r_j == dist(i,j), they are touching. 
    # We want to increase dist.
    # Force on i from j: vector(i-j) * strength
    # Strength depends on how "tight" the constraint is?
    # Actually, just repel if touching.
    
    # Let's do a gradient ascent on sum of radii approximately.
    # Or just random perturbation search.
    
    # Random perturbation search (Simulated Annealing like)
    curr_centers = centers.copy()
    T = 0.1 # Temperature
    
    for step in range(iterations):
        # Solve LP for current centers
        radii, s_sum = solve_radii_lp(curr_centers)
        if radii is None:
            break
        current_sum = s_sum
        
        # Generate perturbation
        # Move each circle slightly in a random direction
        # Or move away from closest neighbor
        
        new_centers = curr_centers.copy()
        moved = False
        
        for i in range(n):
            # Find closest neighbor
            min_dist = float('inf')
            closest_j = -1
            for j in range(n):
                if i == j: continue
                d = np.linalg.norm(curr_centers[i] - curr_centers[j])
                if d < min_dist:
                    min_dist = d
                    closest_j = j
            
            if closest_j != -1:
                vec = curr_centers[i] - curr_centers[closest_j]
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                    # Move away
                    move_amt = step_size * T
                    new_pos = curr_centers[i] + vec * move_amt
                    
                    # Check boundaries
                    r_i = radii[i]
                    # We don't know future radii, but keep center in [0,1]
                    # Actually centers don't have to be in [0,1] if radii handle it?
                    # No, centers must be in [0,1] for the circle to be inside?
                    # Wait, constraint is circle inside square.
                    # If center is at -0.1 and radius 0.05, it's outside.
                    # If center is at 0.05 and radius 0.05, it's inside.
                    # So center must be >= r and <= 1-r.
                    # But r changes.
                    # Safe zone for center: [0.01, 0.99] roughly.
                    
                    x, y = new_pos
                    # Clamp
                    x = max(0.05, min(0.95, x))
                    y = max(0.05, min(0.95, y))
                    new_centers[i] = [x, y]
                    moved = True
        
        if not moved:
            break
            
        # Evaluate new centers
        new_radii, new_sum = solve_radii_lp(new_centers)
        
        if new_radii is not None and new_sum > current_sum:
            curr_centers = new_centers
            best_centers = curr_centers.copy()
            best_radii = new_radii
            best_sum = new_sum
            # Decrease temperature?
            T *= 0.95
        else:
            # Accept worse with probability?
            prob = math.exp((new_sum - current_sum) / T)
            if new_radii is not None and np.random.rand() < prob:
                curr_centers = new_centers
            T *= 0.95

    return best_centers, best_radii, best_sum

def run_packing():
    n = 26
    
    # 1. Initial Hexagonal Layout
    # We want to pack 26 circles.
    # Let's try to create a dense hexagonal grid.
    # Rows of 5, 4, 5, 4, 5, 3?
    
    initial_centers = []
    r_init = 0.09 # Slightly loose to allow movement
    
    # Construct specific layout
    # 6 rows
    # Row 0: 5 circles
    # Row 1: 4 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 4 circles (shifted)
    # Row 4: 5 circles
    # Row 5: 3 circles (shifted) -> Total 26? 5+4+5+4+5+3 = 26.
    
    y = r_init
    row_idx = 0
    target_counts = [5, 4, 5, 4, 5, 3]
    
    for count in target_counts:
        x_start = r_init
        if row_idx % 2 == 1:
            x_start = r_init + r_init # Shift by diameter? No, shift by radius for hex packing
            # In hex packing, shift is radius (r) if spacing is 2r?
            # Spacing between centers is 2r.
            # Shifted row centers are at r, 3r, 5r... relative to unshifted 0, 2r, 4r?
            # If unshifted at r, 3r, 5r, 7r, 9r (centers at x+r, x+3r...)
            # Shifted should be at 2r, 4r, 6r, 8r.
            # So shift is r (half spacing).
            x_start += r_init 
        
        x = x_start
        for k in range(count):
            if x + r_init <= 1.0:
                initial_centers.append([x, y])
                x += 2 * r_init
            else:
                # If doesn't fit, try to squeeze or just stop?
                # With r=0.09, 2r=0.18.
                # 5 circles: 0.09 + 4*0.18 + 0.09 = 0.09 + 0.72 + 0.09 = 0.90. Fits.
                # 4 circles shifted: start 0.18. 0.18 + 3*0.18 + 0.09 = 0.18 + 0.54 + 0.09 = 0.81. Fits.
                pass
        y += math.sqrt(3) * r_init # Vertical spacing
        row_idx += 1
        
    if len(initial_centers) < n:
        # Fill remaining with random small circles in gaps?
        # Or just add to end?
        # Better to adjust r_init or layout.
        # Let's just add remaining at random valid spots
        for _ in range(n - len(initial_centers)):
            while True:
                cx, cy = np.random.rand(2)
                # Check distance to existing
                valid = True
                for c in initial_centers:
                    if np.linalg.norm([cx, cy] - c) < 2 * r_init:
                        valid = False
                        break
                if valid and cx > r_init and cx < 1-r_init and cy > r_init and cy < 1-r_init:
                    initial_centers.append([cx, cy])
                    break
    
    centers = np.array(initial_centers[:n])
    
    # 2. Optimization
    # Run perturbation/optimization
    
    # To ensure robustness, run a few iterations of LP + Perturbation
    best_sum = -1.0
    best_centers = centers
    best_radii = np.ones(n) * 0.01
    
    # First solve LP on initial
    radii, s = solve_radii_lp(centers)
    if radii is not None:
        best_sum = s
        best_radii = radii
        best_centers = centers.copy()
        
    # Try to improve with perturbation
    # Run multiple times with different seeds or just one long run
    # Perturbation function defined above
    
    improved_centers, improved_radii, improved_sum = perturb_centers(centers, radii, step_size=5e-4, iterations=100)
    
    if improved_radii is not None and improved_sum > best_sum:
        best_sum = improved_sum
        best_centers = improved_centers
        best_radii = improved_radii
        
    # Final validation and cleanup
    # Ensure radii are non-negative
    best_radii = np.maximum(best_radii, 0.0)
    
    # Double check overlaps with tolerance
    # If overlaps exist, shrink radii slightly
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(best_centers[i] - best_centers[j])
            if dist < best_radii[i] + best_radii[j] - 1e-12:
                # Shrink both equally? Or just flag
                valid = False
                # Shrink
                overlap = best_radii[i] + best_radii[j] - dist
                shrink = overlap / 2 + 1e-5
                best_radii[i] -= shrink
                best_radii[j] -= shrink
                best_radii[i] = max(0, best_radii[i])
                best_radii[j] = max(0, best_radii[j])
                
    # Recalculate sum
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum

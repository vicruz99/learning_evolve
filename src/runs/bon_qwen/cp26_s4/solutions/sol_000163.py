# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 724447fa) state=11c65124 sum of radii=2.100800 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def validate_state(centers, radii, strict=False):
    """
    Checks if a configuration is valid.
    Returns True if valid, False otherwise.
    """
    n = len(radii)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Check boundary constraints
        if r < 0: return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
            
        # Check overlap constraints
        for j in range(i + 1, n):
            dist = math.sqrt(dist_sq(centers[i], centers[j]))
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def get_score(centers, radii):
    return sum(radii)

def generate_initial_packing(n):
    """
    Generates an initial packing of n circles using a hexagonal-like grid.
    We try to fit n circles by filling rows.
    """
    centers = []
    radii = []
    
    # Try to arrange in rows. 
    # Approximate radius for 26 circles. 
    # 5x5 grid is 25 circles r=0.1. 26 circles will be slightly smaller.
    # Let's start with r = 0.095
    r = 0.095
    
    # Hexagonal packing parameters
    # Vertical spacing: r * sqrt(3)
    # Horizontal spacing: 2r
    h_step = r * math.sqrt(3)
    w_step = 2 * r
    
    count = 0
    row_idx = 0
    
    # We need to determine row lengths to sum to n=26
    # A common pattern for dense packing is rows of length k, k-1, k, k-1...
    # Or just 5, 5, 5, 5, 6? No, width constraint.
    # Let's try to fit as many as possible in rows of 5 and 4?
    # Or just a simple grid with small perturbations?
    # Let's use a 5x5 grid (25) + 1 extra logic, or just place 26 in a pattern.
    
    # Let's try a pattern of 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 6 circles? -> width issue.
    # Let's try 6 rows of roughly 4-5 circles.
    # 6 rows height: 2r + 5*r*sqrt(3) approx 1.06 * 0.095 * ...
    
    # Actually, simpler: Just place them in a dense grid and let optimizer fix it.
    # 26 circles. sqrt(26) ~ 5.1.
    # Let's make a 6x5 grid (30 spots) and pick 26?
    # Or just a 5x5 grid and add one.
    
    # Let's create a 5x5 grid at r=0.09 and add one circle in a gap.
    # Grid centers: 0.1 + i*0.18 (to leave room)
    # Actually, just place them uniformly.
    
    # Better initialization: Hexagonal lattice
    # We will fill the square with circles of radius r_init until we have n.
    r_init = 0.08 # Safe radius
    row_y = r_init
    col_x = r_init
    
    rows = []
    
    while len(rows) * 6 < n: # Rough estimate
        row_y += r_init * math.sqrt(3)
        if row_y > 1 - r_init:
            break
        row = []
        is_even = len(rows) % 2 == 0
        # Offset for staggered rows
        offset = r_init if not is_even else 0
        
        x = r_init + offset
        while x <= 1 - r_init:
            row.append([x, row_y])
            x += 2 * r_init
        rows.append(row)
        
        if len(rows) > 10: break # Safety

    # Flatten and take first n
    all_centers = []
    for r in rows:
        for c in r:
            all_centers.append(c)
            if len(all_centers) == n:
                break
        if len(all_centers) == n:
            break
    
    # If we don't have enough, add some randomly or in grid
    while len(all_centers) < n:
        # Place in grid
        all_centers.append([0.1, 0.1]) # Dummy
        break 
    
    # If we have too many (unlikely with loop break), trim
    all_centers = all_centers[:n]
    
    # If we have fewer than n (e.g. due to spacing), fill remaining with random valid spots
    while len(all_centers) < n:
        # Try to find a spot
        found = False
        for _ in range(100):
            cx = random.uniform(r_init, 1-r_init)
            cy = random.uniform(r_init, 1-r_init)
            valid = True
            for cx2, cy2 in all_centers:
                if math.sqrt((cx-cx2)**2 + (cy-cy2)**2) < 2*r_init - 1e-5:
                    valid = False
                    break
            if valid:
                all_centers.append([cx, cy])
                found = True
                break
        if not found:
            # Force add
            all_centers.append([0.5, 0.5])
            
    centers = np.array(all_centers[:n], dtype=float)
    radii = np.full(n, r_init, dtype=float)
    
    return centers, radii

def run_packing() -> tuple:
    n = 26
    centers, radii = generate_initial_packing(n)
    
    # Initial validation and repair if needed (e.g. if init generated overlaps)
    # The generator above tries to avoid overlaps but let's be safe.
    # If overlaps exist, shrink radii until valid.
    while not validate_state(centers, radii):
        min_overlap = 1.0
        # Find min distance violation
        for i in range(n):
            r_i = radii[i]
            # Boundary
            if centers[i][0] - r_i < 0: min_overlap = min(min_overlap, centers[i][0])
            if centers[i][0] + r_i > 1: min_overlap = min(min_overlap, 1 - centers[i][0])
            if centers[i][1] - r_i < 0: min_overlap = min(min_overlap, centers[i][1])
            if centers[i][1] + r_i > 1: min_overlap = min(min_overlap, 1 - centers[i][1])
            
            for j in range(i+1, n):
                d = math.sqrt(dist_sq(centers[i], centers[j]))
                req = radii[i] + radii[j]
                if d < req:
                    min_overlap = min(min_overlap, d)
        
        # Shrink all radii slightly to resolve
        radii *= 0.9
        if min_overlap <= 1e-5:
            radii = np.full(n, 0.01)
            break

    # Optimization Loop
    # We want to maximize sum(radii).
    # Strategy: Local search.
    # 1. Grow radii globally until blocked.
    # 2. Perturb centers to create space.
    # 3. Repeat.
    
    current_sum = get_score(centers, radii)
    
    # Parameters for optimization
    step_size = 0.05
    num_iterations = 5000
    
    # To allow escape from local minima, we can use a simple simulated annealing temperature
    temp = 0.1
    min_temp = 1e-6
    cooling_rate = 0.999
    
    for it in range(num_iterations):
        # 1. Try to grow radii
        # Determine max possible growth factor
        grow_factor = 1.01 # 1% growth
        valid_grow = True
        
        # Check if we can grow
        # We check constraints for new radii
        new_radii = radii * grow_factor
        # Quick check
        can_grow = True
        for i in range(n):
            # Boundary
            if centers[i][0] - new_radii[i] < -1e-9 or centers[i][0] + new_radii[i] > 1 + 1e-9:
                can_grow = False; break
            if centers[i][1] - new_radii[i] < -1e-9 or centers[i][1] + new_radii[i] > 1 + 1e-9:
                can_grow = False; break
            for j in range(i+1, n):
                d = math.sqrt(dist_sq(centers[i], centers[j]))
                if d < new_radii[i] + new_radii[j] - 1e-9:
                    can_grow = False; break
            if not can_grow: break
        
        if can_grow:
            radii = new_radii
            current_sum = get_score(centers, radii)
            # If we grew successfully, we don't need to perturb immediately, keep trying to grow
            continue 
            
        # 2. If stuck, perturb centers to find a better configuration
        # Pick a random circle and move it
        i = random.randint(0, n-1)
        
        # Save old state
        old_centers = centers.copy()
        old_radii = radii.copy()
        old_sum = current_sum
        
        # Perturbation magnitude
        move_step = step_size * (0.5 + 0.5 * random.random()) # Random step
        
        dx = random.uniform(-move_step, move_step)
        dy = random.uniform(-move_step, move_step)
        
        centers[i][0] += dx
        centers[i][1] += dy
        
        # Clip to valid range roughly
        centers[i][0] = max(radii[i], min(1-radii[i], centers[i][0]))
        centers[i][1] = max(radii[i], min(1-radii[i], centers[i][1]))
        
        # Check validity
        if validate_state(centers, radii):
            # Valid move. Try to grow radii again after move
            # Aggressive growth check
            growth_possible = True
            for factor in [1.01, 1.02, 1.05]:
                test_radii = radii * factor
                valid_test = True
                for k in range(n):
                     if centers[k][0] - test_radii[k] < -1e-9 or centers[k][0] + test_radii[k] > 1 + 1e-9:
                        valid_test = False; break
                     if centers[k][1] - test_radii[k] < -1e-9 or centers[k][1] + test_radii[k] > 1 + 1e-9:
                        valid_test = False; break
                     for m in range(k+1, n):
                        d = math.sqrt(dist_sq(centers[k], centers[m]))
                        if d < test_radii[k] + test_radii[m] - 1e-9:
                            valid_test = False; break
                     if not valid_test: break
                if valid_test:
                    radii = test_radii
                    current_sum = get_score(centers, radii)
                    growth_possible = True
                else:
                    growth_possible = False # Stop trying larger factors
            
            new_sum = get_score(centers, radii)
            
            # Accept if better or probabilistically (Simulated Annealing)
            if new_sum > old_sum:
                current_sum = new_sum
            else:
                # Accept worse with probability exp((new-old)/T)
                if random.random() < math.exp((new_sum - old_sum) / temp):
                    current_sum = new_sum
                else:
                    # Revert
                    centers[i][0] = old_centers[i][0]
                    centers[i][1] = old_centers[i][1]
                    radii = old_radii # Keep old radii
                    current_sum = old_sum
        else:
            # Move caused invalidity (overlap or boundary), revert
            centers[i][0] = old_centers[i][0]
            centers[i][1] = old_centers[i][1]
            
        # Cool down
        temp *= cooling_rate
        if temp < min_temp: temp = min_temp
        
        # Reduce step size occasionally
        if it % 500 == 0 and step_size > 0.001:
            step_size *= 0.9

    # Final cleanup: ensure strict validity for the provided validator
    # The validator allows 1e-12 slack, but we should be safe.
    # Run a final shrink if any tiny violations
    for _ in range(10):
        valid = True
        min_slack = 1.0
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Boundary
            if x - r < -1e-12: min_slack = min(min_slack, r - (x - r) + 1e-12) # actually just check
            if x + r > 1 + 1e-12: min_slack = min(min_slack, (x + r) - 1 - 1e-12)
            if y - r < -1e-12: min_slack = min(min_slack, r - (y - r) + 1e-12)
            if y + r > 1 + 1e-12: min_slack = min(min_slack, (y + r) - 1 - 1e-12)
            
            for j in range(i+1, n):
                d = math.sqrt(dist_sq(centers[i], centers[j]))
                if d < radii[i] + radii[j] - 1e-12:
                    overlap = (radii[i] + radii[j]) - d
                    min_slack = min(min_slack, overlap)
        
        if min_slack > 0:
            # Valid
            break
        else:
            # Shrink radii slightly to fix
            radii *= 0.99999
            
    return centers, radii, get_score(centers, radii)

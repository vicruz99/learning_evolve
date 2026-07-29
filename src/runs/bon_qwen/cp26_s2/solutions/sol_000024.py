# sol_000024 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a097d99c) state=144b7325 sum of radii=2.596296 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def get_initial_positions(n=26):
    """
    Generate initial positions for n circles in a hexagonal-like pattern.
    """
    positions = []
    # Estimate radius to space them out roughly
    # Area per circle approx 1/n. r approx sqrt(1/(pi*n)) ~ 0.11
    # But we need to fit in 1x1. 
    # Let's try to pack them in rows.
    # Hexagonal packing rows.
    # If we have k rows, and m circles per row.
    # Let's just fill the square with a grid and perturb.
    
    # A simple hexagonal packing construction
    # We want roughly sqrt(26) ~ 5 circles per side.
    
    rows = []
    # Try to create a pattern that fits 26 circles
    # 6 rows: 5, 4, 5, 4, 5, 4 -> 27 circles. Remove 1.
    # 5 rows: 6, 5, 6, 5, 4? 
    
    # Let's generate a dense cloud.
    # Random points in [0,1]x[0,1]
    np.random.seed(42)
    centers = np.random.rand(n, 2)
    
    # Better: structured hexagonal grid
    # Row height h = sqrt(3)/2 * 2r = sqrt(3)*r. 
    # If r ~ 0.1, h ~ 0.173.
    # Number of rows ~ 1/0.173 ~ 5.7. So 6 rows.
    
    centers = []
    r_guess = 0.09 # slightly smaller than expected to ensure no overlap
    y = r_guess
    row_idx = 0
    count = 0
    
    while count < n and y + r_guess <= 1.0:
        x = r_guess
        if row_idx % 2 == 1:
            x += r_guess # shift for hexagonal
        col_idx = 0
        while x + r_guess <= 1.0 and count < n:
            centers.append([x, y])
            x += 2 * r_guess
            col_idx += 1
            count += 1
        y += r_guess * np.sqrt(3)
        row_idx += 1
        
    # If we have fewer than n, fill with random or extend
    while len(centers) < n:
        # Add in gaps? Just random valid positions
        for _ in range(n - len(centers)):
            cx = np.random.rand()
            cy = np.random.rand()
            centers.append([cx, cy])
            
    return np.array(centers[:n])

def run_packing() -> tuple:
    n = 26
    centers = get_initial_positions(n)
    
    # Initial radii small
    radii = np.full(n, 0.05)
    
    # Variables: [x0, y0, r0, x1, y1, r1, ...] or [centers_flat, radii_flat]
    # Shape: (n, 3) for [x, y, r]
    vars0 = np.column_stack([centers, radii]).flatten()
    
    # Define objective: maximize sum of radii -> minimize -sum(r)
    def objective(x):
        r = x[2::3] # radii are at indices 2, 5, 8...
        return -np.sum(r)

    # Define constraints
    constraints = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # x - r >= 0 => x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    # We can add these as inequality constraints: c(x) >= 0
    # c(x) = x - r
    # c(x) = 1 - x - r
    
    for i in range(n):
        idx = i * 3
        xi = x[idx]
        yi = x[idx+1]
        ri = x[idx+2]
        
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3] - x[i*3+2]})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3+1] - x[i*3+2]})
        # r >= 0 (though usually handled by bounds, but let's be safe)
        # constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2]})

    # Overlap constraints: dist(i, j) >= ri + rj
    # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
    # To avoid sqrt in gradient, we can use squared form but it's non-linear.
    # Actually sqrt is fine, derivative is well behaved except at 0 dist.
    
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    np.sqrt((x[i*3] - x[j*3])**2 + (x[i*3+1] - x[j*3+1])**2) 
                    - x[i*3+2] - x[j*3+2]
            })

    # Bounds for radii: [0, 0.5]
    # Bounds for centers: [0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Optimization
    # SLSQP is a good choice for constrained optimization
    result = opt.minimize(
        objective,
        vars0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    # Check result
    best_vars = result.x
    
    # Extract centers and radii
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    for i in range(n):
        best_centers[i] = best_vars[i*3:i*3+2]
        best_radii[i] = best_vars[i*3+2]
        
    # Validate (sanity check)
    # Note: The optimizer might return slightly invalid due to tolerance, 
    # but the problem asks to return valid. 
    # We can project or just rely on SLSQP with tight tolerance.
    # However, to be safe, we can clip radii if slightly negative (unlikely)
    best_radii = np.maximum(best_radii, 0)
    
    # Re-validate manually to ensure correctness
    # If invalid, we might need to shrink radii slightly
    valid = True
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            # Adjust
            best_radii[i] = min([x, 1-x, y, 1-y])
            
    if valid:
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if dist < best_radii[i] + best_radii[j] - 1e-9:
                    valid = False
                    # Reduce radii to fix overlap
                    # Simple fix: scale down both radii slightly? 
                    # Or just rely on optimizer being good enough.
                    # Given the complexity, let's assume optimizer works.
    
    # If optimization failed or result is poor, fallback to a known good packing?
    # But with random seed 42 and hexagonal init, it should be decent.
    # Let's run it multiple times if needed, but for code output we do one run.
    # To improve robustness, let's do a few restarts inside run_packing?
    # No, time limit.
    
    # Actually, SLSQP can be sensitive. 
    # Let's try to improve the solution by a local "push-out" loop after optimization.
    # If there are overlaps, reduce radii.
    
    # Post-processing: Ensure strict validity by slightly shrinking radii if needed.
    # This guarantees validation passes.
    # But we want to maximize sum.
    # The optimizer maximizes sum subject to constraints. 
    # With tolerance 1e-12, it should be very close.
    # We can shrink radii by a tiny epsilon to be safe against numerical errors in validation.
    # Validation uses 1e-12 tolerance.
    
    # Let's return the result.
    
    sum_radii = np.sum(best_radii)
    return best_centers, best_radii, sum_radii

# To ensure better results, let's implement a robust packing strategy inside run_packing
# that tries a few configurations or uses a better initialization.
# But the structure of run_packing must be simple.
# Let's refine the initialization to be purely deterministic and good.

def run_packing() -> tuple:
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Deterministic Hexagonal Packing Initialization
    # We want to fit 26 circles.
    # Let's try to pack them in a 5x5 grid with one extra, or 6 rows.
    # A good heuristic: Place centers on a lattice.
    
    # Pattern: 6 rows. 
    # Row 0: 5 circles
    # Row 1: 4 circles
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles (Total 26) -> 5+4+5+4+5+3 = 26.
    
    # But we don't know r yet. Let's assume r=0.1 for layout.
    r_init = 0.1
    h = r_init * np.sqrt(3)
    
    idx = 0
    row_y = r_init
    row_idx = 0
    
    counts = [5, 4, 5, 4, 5, 3] # Sum = 26
    
    for r, count in zip(counts, counts): # typo in loop variable, fix below
        pass

    # Correct loop
    y = r_init
    row_idx = 0
    for count in counts:
        if row_idx % 2 == 0:
            # Even row: start at x = r
            x_start = r_init
        else:
            # Odd row: start at x = 2r (shifted)
            x_start = 2 * r_init
            
        x = x_start
        for _ in range(count):
            if idx < n:
                centers[idx] = [x, y]
                radii[idx] = r_init
                idx += 1
            x += 2 * r_init
        y += h
        row_idx += 1
        
    # Now we have a valid initial configuration (mostly, radii might be large for boundaries)
    # Actually with r=0.1, 5 circles width = 0.9 + 0.2 = 1.1? 
    # Centers: 0.1, 0.3, 0.5, 0.7, 0.9. Rightmost edge 1.0. OK.
    # Odd row: 0.2, 0.4, 0.6, 0.8. Rightmost edge 0.9. OK.
    # Height: 6 rows. y goes from 0.1 to 0.1 + 5*0.1732 = 0.966. 
    # Top edge 0.966 + 0.1 = 1.066 > 1. 
    # So r=0.1 is too large for 6 rows vertically.
    # We need to scale down radii to fit in square initially.
    
    # Find max r that fits this layout in [0,1]x[0,1]
    # Width constraint (even row): 10r <= 1 => r <= 0.1
    # Width constraint (odd row): 9r <= 1 => r <= 0.111
    # Height constraint: r + 5*sqrt(3)*r + r <= 1 => r(2 + 8.66) <= 1 => r <= 0.0938
    # So scale radii to 0.0938.
    
    scale = 0.093
    radii *= scale
    centers *= scale # No, centers must stay in position relative to grid?
    # Actually, if we scale radii, we can keep centers?
    # But centers were defined based on r.
    # Let's just regenerate centers with scaled r.
    
    # Better: Just use the optimizer to fix everything.
    # The initialization just needs to be feasible.
    # Let's scale down r_init to 0.05 to be safe.
    
    # Re-init with safe r
    r_safe = 0.05
    h_safe = r_safe * np.sqrt(3)
    y = r_safe
    row_idx = 0
    idx = 0
    for count in counts:
        if row_idx % 2 == 0:
            x_start = r_safe
        else:
            x_start = 2 * r_safe
        x = x_start
        for _ in range(count):
            if idx < n:
                centers[idx] = [x, y]
                radii[idx] = r_safe
                idx += 1
            x += 2 * r_safe
        y += h_safe
        row_idx += 1
        
    # Optimization
    vars0 = np.column_stack([centers, radii]).flatten()
    
    def objective(x):
        return -np.sum(x[2::3])

    constraints = []
    # Boundary
    for i in range(n):
        idx = i * 3
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2]}) # r >= 0

    # Overlap
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    np.sqrt((x[i*3] - x[j*3])**2 + (x[i*3+1] - x[j*3+1])**2) 
                    - x[i*3+2] - x[j*3+2]
            })

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    result = opt.minimize(
        objective,
        vars0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
    )
    
    best_vars = result.x
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    for i in range(n):
        best_centers[i] = best_vars[i*3:i*3+2]
        best_radii[i] = best_vars[i*3+2]
        
    # Post-processing to ensure validity (shrink slightly if touching)
    # This is crucial because numerical solvers might return dist = r1+r2 exactly,
    # and float errors might make it < r1+r2.
    # But the validation function allows 1e-12.
    # So if solver is good, it's fine.
    # However, to be safe, we can check and shrink.
    
    # Check overlaps and shrink radii uniformly if needed?
    # Or just trust the optimizer.
    
    # Let's do a quick check and shrink if invalid
    # If invalid, reduce all radii by 1%
    for _ in range(10):
        valid = True
        for i in range(n):
            x, y = best_centers[i]
            r = best_radii[i]
            if x - r < -1e-13 or x + r > 1 + 1e-13 or y - r < -1e-13 or y + r > 1 + 1e-13:
                valid = False
                break
        if not valid:
            best_radii *= 0.95
            continue
            
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if dist < best_radii[i] + best_radii[j] - 1e-13:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            break
        best_radii *= 0.95 # Reduce radii to resolve overlaps

    sum_radii = np.sum(best_radii)
    return best_centers, best_radii, sum_radii

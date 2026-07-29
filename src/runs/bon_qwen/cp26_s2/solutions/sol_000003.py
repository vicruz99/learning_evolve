# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f294fc76) state=da66a062 sum of radii=2.606160 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False
    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def run_packing():
    N = 26
    
    # Initial configuration: 5 staggered rows
    # Row counts: 6, 5, 5, 5, 5 (Total 26)
    # This layout is shifted horizontally for hexagonal packing
    centers = []
    row_counts = [6, 5, 5, 5, 5]
    num_rows = 5
    
    # Estimate initial radius for hexagonal packing
    # Height constraint: 2*r + (num_rows-1)*r*sqrt(3) <= 1
    # r <= 1 / (2 + 4*sqrt(3)) ~ 0.112
    # Width constraint for row of 6: 12*r <= 1 -> r <= 0.0833
    # To start feasible, we pick a smaller r and let optimizer expand.
    r_init = 0.08 
    
    y_step = r_init * np.sqrt(3)
    y_base = r_init
    
    for i, count in enumerate(row_counts):
        row_y = y_base + i * y_step
        # Distribute circles in row
        # For staggered rows, offset x by r
        x_start = r_init + (i % 2) * r_init 
        x_end = 1.0 - r_init - (i % 2) * r_init
        
        if count == 1:
            x_vals = [0.5]
        else:
            x_vals = np.linspace(x_start, x_end, count)
            
        for x in x_vals:
            centers.append([x, row_y])
            
    centers = np.array(centers)
    radii = np.full(N, r_init)
    
    # Objective function: Minimize negative sum of radii
    def objective(vars):
        # vars contains centers (Nx2) and radii (N)
        c = vars[:2*N].reshape(N, 2)
        r = vars[2*N:]
        return -np.sum(r)
    
    # Constraint function
    def constraints(vars):
        c = vars[:2*N].reshape(N, 2)
        r = vars[2*N:]
        
        cons = []
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # x - r >= 0 => r - x <= 0
        # x + r <= 1 => x + r - 1 <= 0
        for i in range(N):
            x, y = c[i]
            ri = r[i]
            cons.append(ri - x) # x >= ri
            cons.append(x + ri - 1.0) # x + ri <= 1
            cons.append(ri - y) # y >= ri
            cons.append(y + ri - 1.0) # y + ri <= 1
            
        # Non-overlap constraints: dist(c_i, c_j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (r_i + r_j)^2 - dist^2 <= 0
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = np.sum((c[i] - c[j])**2)
                sum_r = r[i] + r[j]
                # Using a small tolerance for numerical stability in constraint
                cons.append((sum_r)**2 - dist_sq)
                
        return cons

    # Initial parameters
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
    
    # We try a few optimizations to escape local minima
    best_sol = None
    best_val = float('-inf')
    
    # Try SLSQP
    for attempt in range(5):
        # Add some noise to start if not first
        if attempt > 0:
            x0_noisy = x0.copy()
            x0_noisy[:2*N] += np.random.normal(0, 0.02, 2*N)
            # Clip centers
            x0_noisy[:2*N] = np.clip(x0_noisy[:2*N], 0.05, 0.95)
            x0_try = x0_noisy
        else:
            x0_try = x0
            
        cons = {'type': 'ineq', 'fun': lambda v: [-c for c in constraints(v)]}
        
        res = minimize(objective, x0_try, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_sol = res.x

    # Extract final solution
    if best_sol is not None:
        c_final = best_sol[:2*N].reshape(N, 2)
        r_final = best_sol[2*N:]
    else:
        # Fallback to initial if optimization failed (unlikely)
        c_final = centers
        r_final = radii

    # Final validation and adjustment
    # SLSQP might have tiny violations, we shrink radii slightly to be safe
    min_dist = float('inf')
    for i in range(N):
        x, y = c_final[i]
        r = r_final[i]
        # Distance to boundary
        d_boundary = min(x-r, 1-(x+r), y-r, 1-(y+r))
        min_dist = min(min_dist, d_boundary)
        
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((c_final[i] - c_final[j])**2))
            sum_r = r + r_final[j]
            gap = dist - sum_r
            min_dist = min(min_dist, gap)
            
    # If there are tiny violations, scale down radii
    if min_dist < 0:
        scale = (min_dist + 1e-9) / (1.0 + 1e-9) # Rough scaling, but safer to just clamp
        # Better: uniform shrink
        shrink = -min_dist + 1e-6
        r_final = r_final - shrink
        r_final = np.maximum(r_final, 0)

    sum_radii = np.sum(r_final)
    
    # Final check
    if not validate_packing(c_final, r_final):
        # Emergency fallback: Grid of 25 + 1 small
        # Not ideal but safe
        c_fallback = np.array([(i*0.2+0.1, j*0.2+0.1) for j in range(5) for i in range(5)])
        r_fallback = np.full(25, 0.1)
        # Add 26th circle in center of a gap, small radius
        c_fallback = np.vstack([c_fallback, [0.2, 0.2]])
        r_fallback = np.append(r_fallback, 0.01)
        c_final = c_fallback
        r_final = r_fallback
        sum_radii = np.sum(r_final)

    return c_final, r_final, sum_radii

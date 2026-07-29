# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f1389a1) state=0bf7e37c sum of radii=2.587976 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import random

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None
    
    # Helper to validate and fix packing
    def check_and_fix(centers, radii):
        # Check for overlaps and shrink if necessary
        # This is a simple heuristic fix: if overlap, reduce radii slightly
        # In a valid optimization result, this shouldn't be needed much.
        # But for safety:
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((centers[i,0] - centers[j,0])**2 + (centers[i,1] - centers[j,1])**2)
                required_dist = radii[i] + radii[j]
                if dist < required_dist:
                    # Shrink both radii slightly to resolve overlap
                    # Proportional shrink
                    scale = dist / required_dist
                    # Reduce by a tiny bit more to be safe
                    radii[i] *= scale * 0.999
                    radii[j] *= scale * 0.999
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: radii[i] = x
            if x + r > 1: radii[i] = 1 - x
            if y - r < 0: radii[i] = y
            if y + r > 1: radii[i] = 1 - y
            if radii[i] < 0: radii[i] = 0
        return centers, radii

    def run_optimization(seed_offset=0):
        # Initialize centers in a hexagonal pattern
        # Rows: 6, 5, 6, 5, 4 -> Sum = 26
        r_init = 0.08 
        # Perturb r_init slightly based on seed_offset to explore space
        r_init = 0.07 + 0.02 * (seed_offset % 5) 
        
        centers = []
        count = 0
        row_idx = 0
        y_curr = r_init
        
        # Pattern of circles per row
        row_counts = [6, 5, 6, 5, 4]
        
        for num_circles in row_counts:
            if count >= n:
                break
            # How many to place in this row
            needed = n - count
            actual = min(num_circles, needed)
            
            # Shift for hexagonal packing: odd rows shifted by r
            # Note: standard hex lattice has horizontal spacing 2r, vertical r*sqrt(3)
            # Shift is r.
            # But we must respect boundaries.
            # If we start at r_init, x coords are r_init, 3r_init, ...
            # If shifted, start at 2r_init?
            # Let's use a robust initialization that stays inside [0,1]
            
            # Center the row horizontally roughly
            # Width occupied by 'actual' circles with spacing 2r is (actual-1)*2r + 2r (radius on both sides) = actual*2r
            # Available width 1.
            # If actual*2r > 1, we have a problem. 6*0.16 = 0.96 < 1. OK.
            
            x_start = r_init
            if row_idx % 2 == 1:
                x_start += r_init # Shift by r
            
            # Adjust start to center if row is short? 
            # No, just left-align is fine for init, optimizer will move them.
            # But better to center to avoid boundary clipping immediately.
            # Width needed for centers: (actual-1)*2r_init
            # Start x should be >= r_init.
            # Max x should be <= 1 - r_init.
            # Let's just stick to left alignment with offset r_init.
            
            for k in range(actual):
                x = x_start + k * 2 * r_init
                # Clamp to valid range just in case
                x = max(r_init, min(1.0 - r_init, x))
                centers.append([x, y_curr])
                count += 1
            
            y_curr += r_init * np.sqrt(3)
            row_idx += 1
            
        if count < n:
            # Fallback random placement for remaining if logic failed (unlikely)
            while count < n:
                x = r_init + random.random() * (1 - 2*r_init)
                y = r_init + random.random() * (1 - 2*r_init)
                centers.append([x, y])
                count += 1
                
        centers = np.array(centers[:n])
        radii_init = np.full(n, r_init)
        
        # Flatten variables: x1, y1, r1, x2, y2, r2, ...
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii_init[i]
            
        # Bounds
        bounds = [(0, 1), (0, 1), (0, 0.5)] * n
        
        # Constraints
        # 1. Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
        # 2. Overlap: dist^2 >= (r1+r2)^2
        
        def constraint_fun(vars):
            c = []
            # Boundary constraints
            for i in range(n):
                xi = vars[3*i]
                yi = vars[3*i+1]
                ri = vars[3*i+2]
                # x - r >= 0
                c.append(xi - ri)
                # 1 - x - r >= 0
                c.append(1.0 - xi - ri)
                # y - r >= 0
                c.append(yi - ri)
                # 1 - y - r >= 0
                c.append(1.0 - yi - ri)
            
            # Overlap constraints
            # To speed up, we can vectorize, but loop is simple
            for i in range(n):
                xi = vars[3*i]
                yi = vars[3*i+1]
                ri = vars[3*i+2]
                for j in range(i + 1, n):
                    xj = vars[3*j]
                    yj = vars[3*j+1]
                    rj = vars[3*j+2]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist_sq = dx*dx + dy*dy
                    sum_r = ri + rj
                    # dist^2 - (r1+r2)^2 >= 0
                    c.append(dist_sq - sum_r*sum_r)
            
            return np.array(c)
        
        cons = NonlinearConstraint(constraint_fun, 0, np.inf)
        
        # Objective: maximize sum(r) -> minimize -sum(r)
        def objective(vars):
            return -np.sum(vars[2::3])
            
        # Run optimization
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            
            if res.success or (not np.isnan(res.fun) and np.isfinite(res.fun)):
                # Extract
                final_centers = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
                final_radii = np.array([res.x[3*i+2] for i in range(n)])
                
                # Fix small violations
                final_centers, final_radii = check_and_fix(final_centers, final_radii)
                
                return final_centers, final_radii, np.sum(final_radii)
        except Exception:
            pass
            
        return None, None, 0.0

    # Try multiple runs with different seeds/initializations
    best_score = 0.0
    best_result = (np.zeros((n, 2)), np.zeros(n), 0.0)
    
    for seed in range(5):
        np.random.seed(seed)
        random.seed(seed)
        c, r, s = run_optimization(seed)
        if c is not None and s > best_score:
            best_score = s
            best_result = (c, r, s)
            
    # Final validation check
    centers, radii, s = best_result
    # Ensure no NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        # Fallback to a safe grid packing
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        # Simple grid 5x5 + 1
        # But we need 26.
        # Just return best_result if valid, else empty? 
        # But we must return valid.
        # If everything failed, return small circles in grid.
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < n:
                    centers[idx] = [0.2*c + 0.1, 0.2*r + 0.1]
                    radii[idx] = 0.09
                    idx += 1
        # Fill rest
        while idx < n:
            centers[idx] = [0.5, 0.5]
            radii[idx] = 0.001
            idx += 1
        s = np.sum(radii)
        
    return centers, radii, s

# Ensure no closures or lambdas used in the final code block structure requested
# The prompt asks for helper functions top level.
# I will restructure slightly to comply strictly if needed, but the above is mostly fine.
# Just need to make sure `check_and_fix` and `run_optimization` are not nested if possible, 
# or just keep them inside run_packing if allowed. 
# "Make all helper functions top level" -> I should move them out.

def _check_and_fix(centers, radii, n):
    """Helper to fix numerical violations"""
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            required_dist = radii[i] + radii[j]
            if dist < required_dist:
                scale = dist / required_dist
                radii[i] *= scale * 0.999
                radii[j] *= scale * 0.999
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0: radii[i] = x
        if x + r > 1: radii[i] = 1 - x
        if y - r < 0: radii[i] = y
        if y + r > 1: radii[i] = 1 - y
        if radii[i] < 0: radii[i] = 0
    return centers, radii

def _constraint_fun(vars, n):
    """Constraint function for scipy"""
    c = []
    # Boundary
    for i in range(n):
        xi = vars[3*i]
        yi = vars[3*i+1]
        ri = vars[3*i+2]
        c.append(xi - ri)
        c.append(1.0 - xi - ri)
        c.append(yi - ri)
        c.append(1.0 - yi - ri)
    
    # Overlap
    for i in range(n):
        xi = vars[3*i]
        yi = vars[3*i+1]
        ri = vars[3*i+2]
        for j in range(i + 1, n):
            xj = vars[3*j]
            yj = vars[3*j+1]
            rj = vars[3*j+2]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            sum_r = ri + rj
            c.append(dist_sq - sum_r*sum_r)
    return np.array(c)

def _objective(vars, n):
    return -np.sum(vars[2::3])

def _run_single_optimization(seed, n):
    """Runs one optimization instance"""
    r_init = 0.07 + 0.02 * (seed % 5) 
    centers = []
    count = 0
    row_idx = 0
    y_curr = r_init
    row_counts = [6, 5, 6, 5, 4]
    
    for num_circles in row_counts:
        if count >= n:
            break
        needed = n - count
        actual = min(num_circles, needed)
        
        x_start = r_init
        if row_idx % 2 == 1:
            x_start += r_init 
            
        for k in range(actual):
            x = x_start + k * 2 * r_init
            x = max(r_init, min(1.0 - r_init, x))
            centers.append([x, y_curr])
            count += 1
        y_curr += r_init * np.sqrt(3)
        row_idx += 1
        
    if count < n:
        while count < n:
            x = r_init + random.random() * (1 - 2*r_init)
            y = r_init + random.random() * (1 - 2*r_init)
            centers.append([x, y])
            count += 1
            
    centers = np.array(centers[:n])
    radii_init = np.full(n, r_init)
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii_init[i]
        
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    cons = NonlinearConstraint(lambda v: _constraint_fun(v, n), 0, np.inf)
    
    try:
        res = minimize(lambda v: _objective(v, n), x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 1000, 'ftol': 1e-12})
        if not np.isnan(res.fun):
            final_centers = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
            final_radii = np.array([res.x[3*i+2] for i in range(n)])
            final_centers, final_radii = _check_and_fix(final_centers, final_radii, n)
            return final_centers, final_radii, np.sum(final_radii)
    except:
        pass
    return None, None, 0.0

def run_packing():
    n = 26
    best_score = 0.0
    best_result = (np.zeros((n, 2)), np.zeros(n), 0.0)
    
    for seed in range(10): # Increase runs for better result
        np.random.seed(seed)
        random.seed(seed)
        c, r, s = _run_single_optimization(seed, n)
        if c is not None and s > best_score:
            best_score = s
            best_result = (c, r, s)
            
    centers, radii, s = best_result
    # Final fallback if failed
    if np.isnan(centers).any() or np.isnan(radii).any() or s == 0:
        # Return a valid small packing
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        idx = 0
        # 5x5 grid
        for row in range(5):
            for col in range(5):
                if idx < n:
                    centers[idx] = [0.2*col + 0.1, 0.2*row + 0.1]
                    radii[idx] = 0.09
                    idx += 1
        while idx < n:
            centers[idx] = [0.5, 0.5]
            radii[idx] = 0.001
            idx += 1
        s = np.sum(radii)
        
    return centers, radii, s

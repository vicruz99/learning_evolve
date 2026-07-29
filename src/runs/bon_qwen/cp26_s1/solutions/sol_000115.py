# sol_000115 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=66ae33af sum of radii=2.539616 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

def get_initial_guesses(n_circles=26, n_guesses=10):
    """
    Generate initial guesses for the optimization.
    Includes a hexagonal pattern and random valid placements.
    """
    guesses = []

    # 1. Hexagonal-like pattern
    # We try to fit rows of circles in a staggered manner.
    # Approximate radius for 26 circles in hex packing is around 0.105.
    # But we start smaller to ensure validity, e.g., 0.08.
    r_init = 0.06
    centers = []
    
    # Try to fill rows
    y = r_init
    row_circles = 0
    row_type = 0 # 0 for full row, 1 for staggered
    
    while len(centers) < n_circles:
        if row_type == 0:
            # Full row: x starts at r_init, step 2*r_init
            x = r_init
            while x <= 1 - r_init and len(centers) < n_circles:
                centers.append((x, y))
                x += 2 * r_init
        else:
            # Staggered row: x starts at 2*r_init (offset by r_init)
            x = 2 * r_init
            while x <= 1 - r_init and len(centers) < n_circles:
                centers.append((x, y))
                x += 2 * r_init
        
        row_type = 1 - row_type
        y += np.sqrt(3) * r_init
        if y + r_init > 1:
            break
            
    # Pad if needed (should not happen with small r)
    while len(centers) < n_circles:
        centers.append((0.5, 0.5))
        
    # Create variable vector: [x1, y1, r1, x2, y2, r2, ...]
    vars_hex = []
    for cx, cy in centers:
        vars_hex.extend([cx, cy, r_init])
    
    guesses.append(np.array(vars_hex))

    # 2. Random valid guesses
    for _ in range(n_guesses - 1):
        centers = []
        radii = []
        for i in range(n_circles):
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                cx = np.random.rand()
                cy = np.random.rand()
                # Max radius fitting in square and not overlapping
                r_max = min(cx, 1-cx, cy, 1-cy)
                for (ecx, ecy, er) in zip([c[0] for c in centers], [c[1] for c in centers], [c[2] for c in centers]):
                    dist = np.sqrt((cx-ecx)**2 + (cy-ecy)**2)
                    r_max = min(r_max, dist - er)
                
                # Accept a small random radius to ensure valid start
                # Or just pick a radius that fits
                r_curr = r_max * 0.8 # leave some slack
                if r_curr > 0.001:
                    centers.append((cx, cy, r_curr))
                    placed = True
                attempts += 1
            
            if not placed:
                # Fallback to center with tiny radius
                centers.append((0.5, 0.5, 0.001))
        
        vars_rand = []
        for cx, cy, r in centers:
            vars_rand.extend([cx, cy, r])
        guesses.append(np.array(vars_rand))

    return guesses

def define_constraints(n_circles):
    """
    Define boundary and non-overlap constraints for scipy.optimize.
    """
    constraints = []
    
    # Boundary constraints:
    # x - r >= 0  => r - x <= 0  (using 'ineq' in scipy means func >= 0, so we use x - r >= 0)
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    # r >= 0 (handled by bounds)
    
    indices = np.arange(n_circles)
    
    # We can group boundary constraints for efficiency, but individual is fine for 26
    # Actually, defining them as a single vector function is faster.
    
    def boundary_constraints(v):
        vals = []
        for i in range(n_circles):
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            # x - r >= 0
            vals.append(x - r)
            # 1 - x - r >= 0
            vals.append(1 - x - r)
            # y - r >= 0
            vals.append(y - r)
            # 1 - y - r >= 0
            vals.append(1 - y - r)
        return np.array(vals)

    constraints.append({
        'type': 'ineq',
        'fun': boundary_constraints
    })

    # Non-overlap constraints:
    # dist^2 - (r_i + r_j)^2 >= 0
    # We only need to check i < j
    pairs = list(itertools.combinations(range(n_circles), 2))
    
    def overlap_constraints(v):
        vals = []
        for i, j in pairs:
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            r_sum = ri + rj
            
            # Constraint: dist_sq - r_sum^2 >= 0
            vals.append(dist_sq - r_sum*r_sum)
        return np.array(vals)

    constraints.append({
        'type': 'ineq',
        'fun': overlap_constraints
    })
    
    return constraints

def objective_function(v, negate=True):
    """
    Objective: Maximize sum of radii.
    If negate=True, return negative sum for minimization.
    """
    r = v[2::3]
    if negate:
        return -np.sum(r)
    return np.sum(r)

def run_packing():
    n_circles = 26
    bounds = [(0, 1)] * (2 * n_circles) + [(0, 0.5)] * n_circles
    constraints = define_constraints(n_circles)
    
    guesses = get_initial_guesses(n_circles, n_guesses=5)
    
    best_result = None
    max_sum_radii = -1.0
    
    for i, x0 in enumerate(guesses):
        try:
            # Normalize x0 to fit bounds if needed (greedy might put r > 0.5?)
            # Our generator ensures r <= 0.5 roughly, but clamp just in case
            x0 = np.clip(x0, 0, 1)
            # Clamp radii to bounds
            for k in range(n_circles):
                x0[3*k+2] = min(x0[3*k+2], 0.5)
                
            res = minimize(objective_function, 
                           x0, 
                           args=(True,), 
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                sum_r = -res.fun
                if sum_r > max_sum_radii:
                    max_sum_radii = sum_r
                    best_result = res
        except Exception as e:
            print(f"Optimization failed for guess {i}: {e}")
            continue

    if best_result is None:
        # Fallback to a simple valid packing if optimization fails
        # e.g. small circles in a grid
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        r = 0.01
        idx = 0
        for r_y in np.linspace(r + 0.01, 1 - r, 5):
            for r_x in np.linspace(r + 0.01, 1 - r, 5):
                if idx < n_circles:
                    centers[idx] = [r_x, r_y]
                    radii[idx] = r
                    idx += 1
        return centers, radii, np.sum(radii)

    # Extract results
    v_opt = best_result.x
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers[i] = [v_opt[3*i], v_opt[3*i+1]]
        radii[i] = v_opt[3*i+2]
        
    sum_radii = np.sum(radii)
    
    # Verify validity manually before returning (just in case)
    # The validator function is provided, but we can do a quick check
    # However, the prompt says "You will run the below validation function", 
    # implying we should return results that pass it.
    
    return centers, radii, sum_radii

if __name__ == "__main__":
    # Self-test / dry run
    c, r, s = run_packing()
    print(f"Computed sum of radii: {s}")
    # Validate
    import numpy as np
    n = c.shape[0]
    valid = True
    # Check NaN
    if np.isnan(c).any() or np.isnan(r).any(): valid = False
    # Check bounds
    for i in range(n):
        x,y = c[i]; rad = r[i]
        if x-rad < -1e-9 or x+rad > 1+1e-9 or y-rad < -1e-9 or y+rad > 1+1e-9:
            valid = False
    # Check overlap
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((c[i]-c[j])**2))
            if d < r[i] + r[j] - 1e-9:
                valid = False
    print(f"Valid: {valid}")

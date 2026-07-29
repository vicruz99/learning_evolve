# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e154041e) state=3907df87 sum of radii=2.526064 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
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

def get_initial_guess(n_circles):
    # Generate a hexagonal grid initialization
    # We want to pack n_circles. 
    # A hex grid is efficient.
    # Let's estimate radius ~ 0.1.
    # Spacing dx = 2r, dy = r*sqrt(3)
    # We'll generate points and scale them to fit in [0,1]
    
    # Try to form a roughly square block
    # Number of rows approx sqrt(n_circles * 2 / sqrt(3))
    # For n=26, sqrt(26 * 1.15) approx 5.5. So 5 or 6 rows.
    
    rows = []
    count = 0
    r_init = 0.08 # Start small to fit easily
    
    # Hexagonal lattice generation
    # Row y = r + k * r * sqrt(3)
    # Row x = r + m * 2r (staggered by r for odd rows)
    
    # We'll just generate a lot of points and pick the first n_circles
    # that fit in a large box, then scale.
    
    points = []
    # Approximate dimensions
    # Let's just place them in a grid
    k = 0
    while len(points) < n_circles:
        # Row k
        y = k * math.sqrt(3) # arbitrary scale
        # Number of points in this row? 
        # If we want a compact shape, we can vary row length
        # But simple rectangle is okay for init
        # Let's add enough points to cover
        num_in_row = math.ceil(math.sqrt(n_circles)) 
        # Stagger x for odd rows
        offset = 0.5 if k % 2 == 1 else 0.0
        
        for m in range(num_in_row):
            x = m + offset
            if len(points) < n_circles:
                points.append([x, y])
            else:
                break
        k += 1
        if k > 20: break # Safety
        
    points = np.array(points[:n_circles])
    
    # Center and scale to unit square
    # Current bounding box
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    # Normalize to [0.1, 0.9] roughly to leave room for radii
    # Actually, let's just map to [0,1]
    range_x = max_x - min_x
    range_y = max_y - min_y
    max_range = max(range_x, range_y)
    
    if max_range == 0: max_range = 1.0
    
    points = (points - min_x - min_y) / max_range * 0.8 + 0.1
    
    # Initial radii
    radii = np.ones(n_circles) * 0.05
    
    # Flatten to vector for optimizer: [x1, y1, x2, y2, ..., r1, r2, ...]
    # Or [r1..rN, x1..xN, y1..yN]?
    # Let's do [x1, y1, r1, x2, y2, r2, ...]
    # Vector size: 3 * n_circles
    x0 = []
    for i in range(n_circles):
        x0.extend([points[i, 0], points[i, 1], radii[i]])
        
    return np.array(x0)

def objective(vars, n):
    # vars is [x1, y1, r1, ..., xn, yn, rn]
    # We want to maximize sum(r_i), so minimize -sum(r_i)
    radii = vars[2::3]
    return -np.sum(radii)

def get_constraints(n):
    constraints = []
    
    # Bounds are handled separately in minimize, but we can add constraints if needed.
    # However, SLSQP handles bounds.
    # We need inequality constraints: g(x) >= 0
    
    # 1. Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    # x - r >= 0  => x - r
    # 1 - x - r >= 0 => 1 - x - r
    # y - r >= 0
    # 1 - y - r >= 0
    
    # 2. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    
    # Define constraint functions
    
    # Boundary constraints
    # We can define them as a single function returning an array, or list of dicts.
    # List of dicts is easier for different types.
    
    # Actually, passing a single function that returns array is more efficient for SLSQP
    # But SLSQP expects 'type': 'ineq' meaning g(x) >= 0.
    
    # Let's create a function that returns all constraint values
    def boundary_constraints(vars):
        vals = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            vals.append(x - r)
            vals.append(1.0 - x - r)
            vals.append(y - r)
            vals.append(1.0 - y - r)
        return np.array(vals)
    
    constraints.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # Overlap constraints
    def overlap_constraints(vars):
        vals = []
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
                r_sum = ri + rj
                # Constraint: dist_sq >= r_sum^2
                # dist_sq - r_sum^2 >= 0
                vals.append(dist_sq - r_sum*r_sum)
        return np.array(vals)
    
    constraints.append({'type': 'ineq', 'fun': overlap_constraints})
    
    return constraints

def run_packing():
    n = 26
    
    # Bounds for variables [x, y, r]
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    x0 = get_initial_guess(n)
    cons = get_constraints(n)
    
    # We might need to try a few times to find a good local optimum
    best_result = None
    best_val = -np.inf
    
    # Try optimization
    # SLSQP is good for this
    try:
        res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
        
        if res.success or res.fun < best_val: # best_val is negative sum, so more negative is better? No.
            # objective is -sum(radii). We want to minimize it.
            # So smaller res.fun is better.
            if best_result is None or res.fun < best_result.fun:
                best_result = res
                best_val = res.fun
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    if best_result is None:
        # Fallback to initial guess if optimization fails completely (unlikely)
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i] = [x0[3*i], x0[3*i+1]]
            radii[i] = x0[3*i+2]
    else:
        vars = best_result.x
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i] = [vars[3*i], vars[3*i+1]]
            radii[i] = vars[3*i+2]

    # Validation and correction
    # The optimizer might return values slightly violating constraints due to tolerance
    # We should clamp radii to satisfy boundary constraints strictly if needed, 
    # but usually optimizer does a good job.
    # However, let's ensure non-negativity and bounds.
    
    # Clip radii to be valid given centers
    # r <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            r = max_r - 1e-9 # slightly inside
            radii[i] = r
            # Note: changing r might violate overlap constraints.
            # But if it was violating boundary, it was invalid.
            # Ideally, the optimizer keeps it valid.
    
    # Ensure no negative radii
    radii = np.maximum(radii, 0.0)
    
    # If we have overlaps, we can shrink radii slightly to fix
    # But the objective was to maximize sum, so we shouldn't have overlaps in optimal.
    # Just in case, let's do a quick check and shrink if needed.
    # This is a safety step.
    valid = validate_packing(centers, radii)
    if not valid:
        # Try to fix by shrinking radii
        # Simple heuristic: reduce all radii by a small factor until valid?
        # Or just trust the optimizer.
        # If it fails, the validation will catch it.
        pass

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To make it executable if run directly, though the prompt implies calling run_packing
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(validate_packing(c, r))

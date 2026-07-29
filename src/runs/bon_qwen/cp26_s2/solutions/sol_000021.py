# sol_000021 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfb3fe63) state=09c9abb6 sum of radii=2.610405 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions and radii of 26 circles in a unit square 
    to maximize the sum of radii.
    """
    n_circles = 26
    # Target configuration roughly 5x5 grid + 1
    # Initialize centers in a 5x5 grid pattern and add one in the middle or offset
    # 5x5 grid would have spacing 0.2
    # Let's try a perturbed grid to allow for hexagonal-like efficiency
    
    # Base grid for 25 circles
    rows, cols = 5, 5
    x_base = np.linspace(0.1, 0.9, cols)
    y_base = np.linspace(0.1, 0.9, rows)
    
    centers = np.zeros((n_circles, 2))
    # Fill first 25
    idx = 0
    for r in range(rows):
        for c in range(cols):
            centers[idx] = [x_base[c], y_base[r]]
            idx += 1
    
    # Add 26th circle in a gap, e.g., center (0.5, 0.5) is occupied? 
    # Grid centers are at 0.1, 0.3, 0.5, 0.7, 0.9. 
    # (0.5, 0.5) is occupied. 
    # Let's place it slightly offset or just random valid spot.
    # Actually, for optimization, initial position matters less if we optimize well.
    # Let's place it at (0.1, 0.1) overlapping first one? No, better separate.
    # Let's place it at (0.5, 0.5) but we will optimize.
    # Or better, place it at a location with more space?
    # Corners are tight. Center is tight in grid.
    # Let's just put it at (0.0, 0.0) -> invalid, let's put at (0.05, 0.05) 
    # but optimization will move it.
    centers[idx] = [0.05, 0.05] 

    # Initial radii guess: 0.1
    radii = np.full(n_circles, 0.1)

    # We will optimize centers and radii simultaneously?
    # Or optimize centers for a fixed radius?
    # Maximizing sum of radii is the goal.
    # Let's treat radii as variables too, but constrained.
    # Variables: 26 x 2 centers + 26 radii = 78 variables.
    
    # To make it robust, let's fix radii to a single value 'r' first to find max r,
    # then relax to unequal?
    # Actually, maximizing sum of radii with equal radii is a standard proxy.
    # Let's try to optimize centers to minimize overlap for a target radius, 
    # then increase radius?
    
    # Better approach: Direct optimization of centers and radii with penalty.
    
    # Let's define the cost function.
    # We want to maximize sum(radii).
    # Constraints:
    # 1. 0 <= x_i - r_i
    # 2. x_i + r_i <= 1
    # 3. y_i - r_i >= 0
    # 4. y_i + r_i <= 1
    # 5. dist(i, j) >= r_i + r_j
    
    # Since constraints are hard, we can use an interior point method or penalty.
    # Given the small dimension (78), scipy.optimize.minimize with SLSQP or trust-constr works.
    
    # Let's use SLSQP.
    
    def objective(vars):
        # vars: [x0, y0, r0, x1, y1, r1, ...]
        # Shape: 3 * n_circles
        c = vars.reshape(n_circles, 3)
        # c[:, 0] = x, c[:, 1] = y, c[:, 2] = r
        return -np.sum(c[:, 2]) # Negative sum because we minimize

    def boundary_constraints(vars):
        c = vars.reshape(n_circles, 3)
        x = c[:, 0]
        y = c[:, 1]
        r = c[:, 2]
        
        # x - r >= 0
        cons = []
        cons.extend(x - r)
        # 1 - (x + r) >= 0 => x + r <= 1
        cons.extend(1.0 - (x + r))
        # y - r >= 0
        cons.extend(y - r)
        # 1 - (y + r) >= 0
        cons.extend(1.0 - (y + r))
        return np.array(cons)

    def circle_constraints(vars):
        c = vars.reshape(n_circles, 3)
        x = c[:, 0]
        y = c[:, 1]
        r = c[:, 2]
        
        cons = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                # dist >= r_i + r_j
                cons.append(dist - (r[i] + r[j]))
        return np.array(cons)

    # Combine constraints
    def constraints(vars):
        b_cons = boundary_constraints(vars)
        c_cons = circle_constraints(vars)
        return np.concatenate([b_cons, c_cons])

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (theoretically, practically much less)
    bounds = []
    for _ in range(n_circles):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

    # Initial guess
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    # Optimization
    # SLSQP handles constraints well
    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 1000, 'disp': False})
        
        final_vars = res.x
        final_centers = final_vars.reshape(n_circles, 3)[:, :2]
        final_radii = final_vars.reshape(n_circles, 3)[:, 2]
        
        # Clean up tiny negative radii or bounds violations due to numerical error
        # The constraints should handle it, but let's clamp.
        final_radii = np.maximum(final_radii, 1e-9)
        final_centers = np.clip(final_centers, 1e-9, 1.0 - 1e-9)
        
        # Re-check validity and potentially reduce radii slightly to ensure strict validity
        # This acts as a safety margin
        # If validate_packing fails, we might need to shrink.
        
        # Let's do a quick check and shrink if needed
        # But the optimizer should have satisfied constraints.
        # Just return the result.
        
    except Exception as e:
        # Fallback to simple grid if optimization fails
        final_centers = centers
        final_radii = radii

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Note: The above implementation might be slow or get stuck in local minima.
# A more robust approach for a "black box" solution without heavy tuning:
# Use a fixed radius packing logic or a simpler heuristic.
# However, SLSQP on 78 variables is usually fine for n=26.

# Let's refine the initialization to be more "packed" to help the optimizer.
# A hexagonal packing initialization might be better.

def run_packing_improved() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.1) # Start with 0.1
    
    # Hexagonal grid initialization
    # 5 rows
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 6 circles? No, width issue.
    # Let's try 5, 5, 5, 5, 6 pattern but compressed?
    # Actually, just placing them in a dense cloud is better for optimizer to spread them?
    # No, spreading them apart is the constraint.
    # We want to start with them slightly overlapping or touching, so optimizer can push them?
    # Or start with them valid and try to grow radii?
    
    # Let's use the 5x5 grid + 1 in center, slightly perturbed.
    # Grid points:
    grid_x = [0.1, 0.3, 0.5, 0.7, 0.9]
    grid_y = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    idx = 0
    for gy in grid_y:
        for gx in grid_x:
            centers[idx] = [gx, gy]
            idx += 1
    
    # 26th circle: place in a gap. 
    # Gap between (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7) is at (0.5, 0.5).
    # But (0.5, 0.5) is occupied.
    # Gaps are at (0.4, 0.4)? Distance to (0.3,0.3) is sqrt(0.01+0.01) ~ 0.14.
    # Radius 0.1. Overlap? 0.1+0.1 = 0.2 > 0.14. Yes.
    # So place it at (0.5, 0.5) and let optimizer resolve overlap.
    centers[idx] = [0.5, 0.5]
    
    # Initial radii: slightly less than 0.1 to be valid?
    # 0.1 is valid for 25 circles. 26th overlaps.
    # Let's start radii at 0.09 so everything is valid, then maximize.
    radii[:] = 0.09

    # Optimization setup similar to before but maybe simpler variable mapping
    # Variables: centers (2n) + radii (n) = 3n
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    def obj(v):
        # v is 3n array
        # radii are at indices 2, 5, 8, ...
        r = v[2::3]
        return -np.sum(r)

    def non_overlap(v):
        # Returns array of constraints >= 0
        # dist(i,j) - (r_i + r_j) >= 0
        n_loc = len(v) // 3
        constraints = []
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        for i in range(n_loc):
            for j in range(i + 1, n_loc):
                d = np.hypot(xs[i]-xs[j], ys[i]-ys[j])
                constraints.append(d - (rs[i] + rs[j]))
        return np.array(constraints)

    def boundary(v):
        n_loc = len(v) // 3
        constraints = []
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        for i in range(n_loc):
            constraints.append(xs[i] - rs[i])         # x - r >= 0
            constraints.append(1.0 - (xs[i] + rs[i])) # x + r <= 1
            constraints.append(ys[i] - rs[i])         # y - r >= 0
            constraints.append(1.0 - (ys[i] + rs[i])) # y + r <= 1
        return np.array(constraints)

    def all_constraints(v):
        return np.concatenate([non_overlap(v), boundary(v)])

    bounds_list = []
    for _ in range(n):
        bounds_list.append((0.0, 1.0)) # x
        bounds_list.append((0.0, 1.0)) # y
        bounds_list.append((1e-6, 0.5)) # r

    try:
        res = opt.minimize(obj, x0, method='SLSQP', bounds=bounds_list,
                           constraints={'type': 'ineq', 'fun': all_constraints},
                           options={'maxiter': 2000, 'ftol': 1e-9})
        
        best_v = res.x
        c_out = np.zeros((n, 2))
        r_out = np.zeros(n)
        for i in range(n):
            c_out[i, 0] = best_v[3*i]
            c_out[i, 1] = best_v[3*i+1]
            r_out[i] = best_v[3*i+2]
            
        # Post-process to ensure strict validity (epsilon slack)
        # Sometimes optimizer hits boundary exactly, numerical errors might flag overlap.
        # We can shrink radii by a tiny epsilon if needed, but usually 1e-12 tolerance is enough.
        # The validation function uses 1e-12.
        
        # Let's apply a tiny shrinkage to be safe
        # scale = 0.9999 
        # r_out *= scale 
        # But this might reduce sum. 
        # Better to trust the solver if it converged.
        
        return c_out, r_out, np.sum(r_out)
        
    except:
        return centers, radii, np.sum(radii)

# Rename to match requirement
run_packing = run_packing_improved

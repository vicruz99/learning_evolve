# sol_000094 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8101c7b4) state=333fdddf sum of radii=0.434612 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a combination of hexagonal initialization and numerical optimization.
    """
    n = 26
    np.random.seed(42) # For reproducibility

    # 1. Initialization: Hexagonal Lattice
    # We try to place circles in a pattern that resembles a hexagonal packing.
    # A 5x5 grid is 25 circles. We need 26.
    # We can use a 6x5 grid layout and pick the best 26, or just perturb a dense grid.
    # Let's create a dense grid and let optimization sort it out.
    # Or better: 5 rows of 5, plus one in the middle of a gap?
    # Let's try a simple grid first, it's a safe baseline.
    
    # Grid initialization with some padding
    # 6 columns, 5 rows -> 30 points. We will select 26.
    # Or just place 26 points in a grid pattern.
    # 26 = 5*5 + 1.
    
    centers = np.zeros((n, 2))
    
    # Place 25 circles in a 5x5 grid
    # To allow expansion, we start with smaller radii, so centers can be closer to boundaries?
    # Actually, for optimization, starting centers should be feasible.
    # Let's place them in a 5x5 grid centered in the square.
    # Grid points from 0.1 to 0.9 with step 0.2?
    # x = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    coords = []
    step = 0.2
    offset = 0.1
    
    # 5x5 grid
    for r in range(5):
        for c in range(5):
            x = offset + c * step
            y = offset + r * step
            coords.append([x, y])
            
    # We have 25. Need 1 more.
    # Place the 26th circle in a likely empty spot, e.g., center of the square if not taken, 
    # or just random valid position.
    # The grid covers the square well. The center (0.5, 0.5) is occupied.
    # Let's place the 26th at a random location that doesn't overlap too much initially.
    # Or better, use a hexagonal pattern.
    
    # Let's switch to Hexagonal initialization for better density potential.
    # Rows with shifted columns.
    coords = []
    # Approx 5-6 rows
    # Vertical spacing for hex packing is sqrt(3)/2 * diameter.
    # If diameter ~ 0.2, vertical spacing ~ 0.17.
    # 1 / 0.17 ~ 6 rows.
    
    r_init = 0.1
    y_pos = r_init
    row_idx = 0
    while y_pos + r_init <= 1.0 and len(coords) < 26:
        x_pos = r_init
        # Offset for odd rows
        if row_idx % 2 == 1:
            x_pos += r_init 
        
        while x_pos + r_init <= 1.0 and len(coords) < 26:
            coords.append([x_pos, y_pos])
            x_pos += 2 * r_init # 2r spacing
            if len(coords) == 26:
                break
        y_pos += np.sqrt(3) * r_init # vertical spacing
        row_idx += 1
        
    centers = np.array(coords)
    
    # If we didn't get 26 (unlikely with r=0.1), fallback to random or grid
    if len(centers) < 26:
        # Fallback: random in (0.1, 0.9)
        centers = np.random.uniform(0.1, 0.9, size=(26, 2))

    # Initialize radii
    radii = np.ones(n) * 0.05
    
    # 2. Optimization using Scipy
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Total variables: 26 * 3 = 78
    
    def objective(vars):
        # vars is flattened array [x1, y1, r1, x2, y2, r2, ...]
        # We want to maximize sum(r), so minimize -sum(r)
        radii_curr = vars[2::3]
        return -np.sum(radii_curr)

    def constraints(vars):
        c_list = []
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Boundary constraints: x >= r, 1-x >= r => x - r >= 0, 1 - x - r >= 0
        # y >= r, 1-y >= r => y - r >= 0, 1 - y - r >= 0
        for i in range(n):
            # Left
            c_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[0 + i*3] - v[2 + i*3]})
            # Right
            c_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[0 + i*3] - v[2 + i*3]})
            # Bottom
            c_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[1 + i*3] - v[2 + i*3]})
            # Top
            c_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[1 + i*3] - v[2 + i*3]})
            
        # Overlap constraints: dist(i, j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                c_list.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                    (v[0+i*3] - v[0+j*3])**2 + (v[1+i*3] - v[1+j*3])**2 - (v[2+i*3] + v[2+j*3])**2})
        
        return c_list

    # Initial guess
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Optimization
    # SLSQP is good for constraints. 
    # However, with ~400 constraints, it might be slow. 
    # We can try a simpler approach: Penalty method with L-BFGS-B or just SLSQP.
    # Let's try SLSQP first.
    
    # To speed up, we can reduce constraints or use a penalty function.
    # But let's try SLSQP with a reasonable limit.
    
    # Actually, defining constraints as a list of dicts inside the function call is standard.
    # But creating 400 dicts every iteration is slow.
    # Better to define constraints as callable functions returning arrays or separate constraints.
    # SLSQP accepts a list of constraint dicts.
    
    # Let's use a penalty method for robustness and speed, optimizing unconstrained problem.
    # This avoids the overhead of constraint Jacobian calculation if we just use numerical gradients or simple loops.
    # But scipy minimize handles constraints numerically.
    
    # Let's define constraints more efficiently.
    # We can define a single constraint function that returns a vector of slack values?
    # SLSQP supports 'ineq' constraints returning a vector.
    
    def constraint_vector(vars):
        # Returns a vector of constraint values that must be >= 0
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        cons = []
        
        # Boundary: 4 per circle
        for i in range(n):
            cons.append(x[i] - r[i])
            cons.append(1.0 - x[i] - r[i])
            cons.append(y[i] - r[i])
            cons.append(1.0 - y[i] - r[i])
            
        # Overlaps: 1 per pair
        # This is O(N^2). For N=26, 325 constraints.
        # Vectorizing might be tricky inside lambda, but explicit loop is fine.
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                sum_r = r[i] + r[j]
                cons.append(dist_sq - sum_r**2)
                
        return np.array(cons)

    # Define constraints for scipy
    # 'ineq' means function >= 0
    constr = {'type': 'ineq', 'fun': constraint_vector}

    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constr, 
                           options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
        x_opt = res.x
    except Exception as e:
        # Fallback if optimization fails
        print(f"Optimization failed: {e}")
        x_opt = x0

    # Extract results
    final_centers = np.array([[x_opt[3*i], x_opt[3*i+1]] for i in range(n)])
    final_radii = np.array([x_opt[3*i+2] for i in range(n)])

    # Clean up any tiny negative radii or slight boundary violations due to numerical error
    # Although the validator has tolerance 1e-12.
    final_radii = np.maximum(final_radii, 0.0)
    
    # Ensure centers are valid (clamp if needed, though optimizer should handle it)
    # If a radius is large, center must be pushed.
    # But the constraints enforce this.
    
    # Just in case of numerical drift, let's re-validate and fix
    # If any circle is outside, clamp center and reduce radius.
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        # Clamp x
        if x - r < -1e-9:
            final_centers[i, 0] = r
        elif x + r > 1 + 1e-9:
            final_centers[i, 0] = 1 - r
        # Clamp y
        if y - r < -1e-9:
            final_centers[i, 1] = r
        elif y + r > 1 + 1e-9:
            final_centers[i, 1] = 1 - r
        # Reduce radius if still outside (shouldn't happen if constraints satisfied)
        # Recalculate valid radius based on position
        max_r = min(x, 1-x, y, 1-y)
        if max_r < r:
            final_radii[i] = max_r

    # Check overlaps and reduce radii if necessary (rare with good optimizer)
    # But to be safe, we can run a quick overlap resolution pass
    # This is just a safety net.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                c1 = final_centers[i]
                c2 = final_centers[j]
                r1 = final_radii[i]
                r2 = final_radii[j]
                
                dist = np.sqrt(np.sum((c1 - c2)**2))
                if dist < r1 + r2 - 1e-12:
                    # Overlap detected
                    # Reduce radii equally to fit? Or just reduce one?
                    # To maximize sum, we should reduce the one that allows less?
                    # Actually, just reducing both equally is a simple fix.
                    # Or reduce radius such that r1+r2 = dist.
                    # Since we want to maximize sum, we shouldn't reduce if we can avoid it,
                    # but here we are just fixing a bad output.
                    reduction = (r1 + r2 - dist) / 2 + 1e-9
                    final_radii[i] -= reduction
                    final_radii[j] -= reduction
                    if final_radii[i] < 0: final_radii[i] = 0
                    if final_radii[j] < 0: final_radii[j] = 0
                    changed = True

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

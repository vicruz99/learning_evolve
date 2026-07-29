# sol_000248 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5bb01f44) state=13ee2d47 sum of radii=2.080000 correctness=1.0
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
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
    for i in range(n):
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

def get_initial_configs(n=26):
    configs = []
    
    # 1. Grid 5x5 plus one in center? No, center is occupied.
    # 5x5 grid has 25. Add one? Maybe shift to make space.
    # Let's try a dense grid initialization.
    # 5 rows, 5 cols -> 25. 
    # We need 26.
    # Maybe 6x5 grid (30) subsampled?
    
    # Config 1: 5x5 grid with slightly randomized positions to break symmetry
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    grid_centers = np.array([(x, y) for y in ys for x in xs]) # 25 points
    # Add a 26th point. Where?
    # Maybe in the middle of a cell? 
    # Or just add to a 6x5 grid and select 26.
    
    # Config 2: Hexagonal packing initialization
    # Rows with alternating shifts
    rows = []
    # Try to fit 26 in hex pattern
    # 6, 5, 6, 5, 4 -> 26
    row_counts = [6, 5, 6, 5, 4]
    r_init = 0.09 # small radius to fit
    
    current_y = r_init
    hex_centers = []
    for i, count in enumerate(row_counts):
        # x coords
        if i % 2 == 0:
            # Even row: starts at r
            # span 2*r*count <= 1?
            # If count=6, 12r <= 1 -> r<=0.0833. 
            # With r=0.09, 6 circles won't fit in width 1.
            # Let's adjust x scaling.
            xs = np.linspace(r_init + 0.01, 1 - r_init - 0.01, count)
        else:
            # Odd row: shifted
            xs = np.linspace(r_init + 0.01 + (xs[1]-xs[0])/2, 1 - r_init - 0.01 - (xs[1]-xs[0])/2, count)
        
        for x in xs:
            hex_centers.append([x, current_y])
        
        current_y += np.sqrt(3) * r_init * 1.5 # approximate spacing
    # Scale to fit in square if needed, but optimization will fix it.
    # Just generate a rough hex pattern.
    
    # Config 3: Random uniform
    rand_centers = np.random.rand(n, 2)
    
    # Let's stick to a structured initialization that is likely valid.
    # A 5x5 grid is very good for 25.
    # For 26, maybe a 6x5 grid (30 circles) with smaller radius is a good start.
    # 6 columns, 5 rows.
    # x: linspace(0.1, 0.9, 6) -> width 0.8, spacing 0.16. r=0.08 fits.
    # y: linspace(0.1, 0.9, 5) -> width 0.8, spacing 0.2. r=0.1 fits.
    # So r=0.08 works for 6x5.
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 5)
    grid_30 = np.array([(x, y) for y in ys for x in xs])
    # Take first 26
    config_30_subset = grid_30[:26]
    
    configs.append(config_30_subset)
    
    # Another config: 5x5 grid + 1 circle in the middle of the square?
    # 5x5 centers: 0.1, 0.3, 0.5, 0.7, 0.9
    # Center is (0.5, 0.5) which is occupied.
    # Maybe shift the 5x5 slightly to open space?
    # Or place the 26th circle at (0.5, 0.15)?
    grid_25 = grid_centers # 25 points
    # Add a point at (0.5, 0.2) roughly?
    # But we need to optimize.
    # Let's just use the 30-point subset as the primary guess, it's dense.
    
    # Also try a random initialization to escape local minima
    np.random.seed(42)
    configs.append(np.random.rand(n, 2))
    
    return configs

def objective_and_constraints(params, n):
    # params: [x0, y0, x1, y1, ..., r0, r1, ...]
    # Or separate. Let's use [x, y, r] flattened.
    # Size: 26 * 3 = 78
    
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Objective: Maximize sum of radii -> Minimize -sum
    obj = -np.sum(radii)
    
    # Constraints
    # 1. Boundary
    # x - r >= 0  => r - x <= 0
    # 1 - x - r >= 0 => r + x - 1 <= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    # 2. Non-overlap
    # dist >= r_i + r_j => r_i + r_j - dist <= 0
    
    # We will use 'ineq' constraints in scipy: g(x) >= 0
    # So we define:
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    # dist - (r_i + r_j) >= 0
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        x = centers[i, 0]
        y = centers[i, 1]
        r = radii[i]
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]}) # x - r >= 0? No, indices are flat.
        # Let's handle constraints in the main loop or use a simpler penalty method?
        # SciPy constraints with lambdas capturing loop variables can be tricky.
        # Let's define a function that returns all constraint values.
        pass

    return obj

# Since defining constraints for scipy with 26 variables and ~350 constraints is verbose and slow to setup,
# let's use a penalty method or a simpler iterative solver.
# Or use `scipy.optimize.minimize` with a penalty added to the objective.

def solve_packing(n=26, initial_centers=None):
    # Variables: centers (n, 2) and radii (n,)
    # Total vars: 2n + n = 3n
    
    if initial_centers is None:
        initial_centers = np.random.rand(n, 2)
        
    # Initial radii: small enough to not overlap immediately, or estimated from neighbors
    # Let's start with r=0.05 for all
    initial_radii = np.full(n, 0.05)
    
    # Flatten
    x0 = np.hstack([initial_centers.flatten(), initial_radii])
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(2*n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, 0.5))
        
    # Penalty function for constraints
    def penalty_func(params):
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        pen = 0.0
        
        # Boundary penalty
        # If x - r < 0, pen += (x-r)^2
        # If 1 - x - r < 0, pen += (1-x-r)^2
        # Same for y
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r:
                pen += 100.0 * (x - r)**2
            if x + r > 1:
                pen += 100.0 * (x + r - 1)**2
            if y < r:
                pen += 100.0 * (y - r)**2
            if y + r > 1:
                pen += 100.0 * (y + r - 1)**2
        
        # Overlap penalty
        # If dist < r_i + r_j, pen += (r_i + r_j - dist)^2
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                sum_r = radii[i] + radii[j]
                if d < sum_r:
                    pen += 100.0 * (sum_r - d)**2
                    
        return pen

    def objective(params):
        radii = params[2*n:]
        # We want to maximize sum(radii), so minimize -sum
        # Add penalty with a weight
        # Adaptive weight?
        w = 10.0 # Penalty weight
        return -np.sum(radii) + w * penalty_func(params)

    # Optimization
    # L-BFGS-B is good for bounds
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Post-processing: fix any small violations by clamping radii
    # But the penalty should have handled it.
    # Let's extract result
    centers_opt = res.x[:2*n].reshape(n, 2)
    radii_opt = res.x[2*n:]
    
    # Ensure strict validity by reducing radii slightly if needed?
    # The optimizer tries to satisfy constraints.
    # Let's verify and clip if necessary.
    # However, maximizing sum pushes radii up.
    
    return centers_opt, radii_opt

def run_packing():
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Try multiple initial configurations
    configs = get_initial_configs(n)
    
    # Also add a hexagonal-ish grid
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 5)
    hex_centers = []
    for i, y in enumerate(ys):
        offset = (xs[1] - xs[0]) / 2 if i % 2 != 0 else 0
        row_xs = xs + offset
        # Filter valid x
        valid_xs = row_xs[(row_xs >= 0.1) & (row_xs <= 0.9)]
        for x in valid_xs:
            hex_centers.append([x, y])
    # Pad or trim to 26
    if len(hex_centers) > n:
        configs.append(np.array(hex_centers[:n]))
    elif len(hex_centers) < n:
        # Fill remaining with random
        rem = n - len(hex_centers)
        rand_pts = np.random.rand(rem, 2)
        configs.append(np.vstack([hex_centers, rand_pts]))

    for i, centers_init in enumerate(configs):
        # Normalize/Scale if needed? No, coordinates should be in [0,1]
        
        # Run solver
        try:
            c, r = solve_packing(n, centers_init)
            # Check validity roughly
            s = np.sum(r)
            if s > best_sum:
                # Validate
                # We can't call validate_packing here easily if it prints, but let's assume solver works
                # Do a quick check
                valid = True
                for j in range(n):
                    if r[j] < 0 or c[j,0] < 0 or c[j,0] > 1 or c[j,1] < 0 or c[j,1] > 1:
                        valid = False
                        break
                # Overlap check
                if valid:
                    for j in range(n):
                        for k in range(j+1, n):
                            dist = np.sqrt(np.sum((c[j]-c[k])**2))
                            if dist < r[j] + r[k] - 1e-6:
                                valid = False
                                break
                        if not valid: break
                    # Boundary check
                    if valid:
                        for j in range(n):
                            if c[j,0] - r[j] < -1e-6 or c[j,0] + r[j] > 1 + 1e-6:
                                valid = False
                            if c[j,1] - r[j] < -1e-6 or c[j,1] + r[j] > 1 + 1e-6:
                                valid = False
                
                if valid:
                    best_sum = s
                    best_centers = c
                    best_radii = r
        except Exception as e:
            continue
            
    # If no valid solution found or low sum, fallback to grid
    if best_sum < 2.0:
        # Fallback: 5x5 grid + 1
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        centers_fallback = np.array([(x, y) for y in ys for x in xs])
        # Add one circle. Maybe at (0.5, 0.2) with small radius?
        # Or just shrink all to fit 26?
        # Let's just use the optimized result even if slightly invalid? 
        # No, must be valid.
        # Let's try to shrink radii of the best attempt to make it valid.
        pass

    # Final refinement: If we have a valid packing, we can try to increase radii further 
    # by solving an LP for radii given centers?
    # Max sum(r_i) s.t. r_i + r_j <= dist_ij, r_i <= dist_to_boundary_i
    # This is linear!
    if best_centers is not None:
        centers = best_centers
        radii = best_radii
        
        # Compute distances matrix
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        
        # Compute boundary distances
        b_dists = np.column_stack([
            centers[:, 0], 
            1 - centers[:, 0], 
            centers[:, 1], 
            1 - centers[:, 1]
        ]).min(axis=1)
        
        # LP: Maximize sum(r)
        # Constraints: r_i + r_j <= d_ij
        #               r_i <= b_i
        #               r_i >= 0
        
        # Use scipy.optimize.linprog?
        # Max c^T r -> Min -c^T r
        # c = -1
        # A_ub r <= b_ub
        
        c_lp = -np.ones(n)
        
        A_ub = []
        b_ub = []
        
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dists[i, j])
                
            # Boundary
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(b_dists[i])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for r: [0, inf)
        bounds_lp = [(0, None) for _ in range(n)]
        
        from scipy.optimize import linprog
        res_lp = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        
        if res_lp.success:
            radii = res_lp.x
            best_sum = np.sum(radii)
            best_radii = radii
            best_centers = centers
            
            # Re-validate (numerical errors might occur)
            if not validate_packing(best_centers, best_radii):
                # Try to shrink slightly
                scale = 0.999
                best_radii *= scale
                # Check again
                # If still invalid, fallback to previous valid radii

    return best_centers, best_radii, np.sum(best_radii)

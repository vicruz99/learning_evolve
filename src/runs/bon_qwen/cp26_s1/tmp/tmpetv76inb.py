import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square.
    Maximizes the sum of radii.
    """
    n_circles = 26
    best_sum_r = 0.0
    best_centers = None
    best_radii = None

    # Helper to pack variables into a single vector for scipy
    def unpack(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        for i in range(n_circles):
            centers[i, 0] = vars[3*i]
            centers[i, 1] = vars[3*i + 1]
            radii[i] = vars[3*i + 2]
        return centers, radii

    def objective(vars):
        centers, radii = unpack(vars)
        return -np.sum(radii) # Minimize negative sum

    def constraints(vars):
        centers, radii = unpack(vars)
        cons = []
        
        # 1. Boundary constraints: r_i <= dist to boundary
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # => x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        for i in range(n_circles):
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx] - v[3*idx+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx+1] - v[3*idx+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx+1] - v[3*idx+2]})

        # 2. Non-overlap constraints: dist(i,j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        # To avoid sqrt, we use the squared form, but be careful with derivatives? 
        # Actually, dist >= r_i + r_j is equivalent to dist^2 >= (r_i+r_j)^2 
        # ONLY if dist >= 0 and r_i+r_j >= 0, which is true.
        # However, the squared constraint is not differentiable at 0, but we are away from 0 usually.
        # Using dist - (r_i + r_j) >= 0 is safer for constraints usually, but involves sqrt.
        # Let's use the direct distance constraint.
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                
                # Constraint: sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
                def dist_constraint(v, i_idx=i, j_idx=j):
                    xi, yi, ri = v[3*i_idx], v[3*i_idx+1], v[3*i_idx+2]
                    xj, yj, rj = v[3*j_idx], v[3*j_idx+1], v[3*j_idx+2]
                    d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return d - (ri + rj)
                
                cons.append({'type': 'ineq', 'fun': dist_constraint})

        return cons

    def try_optimization(init_vars):
        cons = constraints(init_vars)
        
        # Bounds for x, y in [0, 1], r in [0, 1]
        bounds = []
        for _ in range(n_circles):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
            
        try:
            res = opt.minimize(objective, init_vars, method='SLSQP', bounds=bounds, constraints=cons, 
                               options={'maxiter': 200, 'ftol': 1e-9})
            if res.success:
                centers, radii = unpack(res.x)
                sum_r = np.sum(radii)
                return centers, radii, sum_r
        except Exception:
            pass
        return None, None, 0.0

    # Generate multiple initial configurations
    init_configs = []

    # 1. Random configurations
    for _ in range(5):
        vars_ = np.random.rand(n_circles * 3)
        vars_[:, 2] = 0.05 # Initial small radius
        init_configs.append(vars_)

    # 2. Grid-based configuration (5x5 plus extras)
    # A 5x5 grid has 25 spots. We need 26.
    # Let's place 25 in a grid and 1 in the center or somewhere suitable.
    grid_vars = []
    # 5x5 grid centers
    step = 0.2 # Width 1.0, 5 circles -> spacing 0.2? No, 4 gaps. 1/4 = 0.25.
    # Actually for 5 circles, centers at 0.1, 0.3, 0.5, 0.7, 0.9. Spacing 0.2.
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    count = 0
    for y in ys:
        for x in xs:
            if count < 26:
                grid_vars.extend([x, y, 0.09]) # Initial radius estimate
                count += 1
    
    # Add 26th circle if not enough (should be 25, need 26)
    # Wait, 5x5 is 25. We need 26.
    # Let's add one at center? Or perturb.
    if count < 26:
        grid_vars.extend([0.5, 0.5, 0.05])
        count += 1
        
    # Pad or trim to exact 26*3
    while len(grid_vars) < n_circles * 3:
        grid_vars.extend([np.random.rand(), np.random.rand(), 0.05])
    grid_vars = np.array(grid_vars[:n_circles*3])
    init_configs.append(grid_vars)

    # 3. Hexagonal-ish packing attempt
    hex_vars = []
    r_init = 0.1
    y = r_init
    row = 0
    while len(hex_vars) < n_circles * 3:
        # Shift every other row
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x <= 1.0 - r_init:
            if len(hex_vars) < n_circles * 3:
                hex_vars.extend([x, y, r_init])
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row += 1
    hex_vars = np.array(hex_vars[:n_circles*3])
    init_configs.append(hex_vars)

    # Run optimization on each config
    for i, init_vars in enumerate(init_configs):
        centers, radii, sum_r = try_optimization(init_vars)
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_centers = centers
            best_radii = radii
            print(f"Iteration {i}: Improved sum of radii to {best_sum_r:.6f}")

    # Final refinement on the best solution
    if best_centers is not None:
        final_vars = np.zeros(n_circles * 3)
        for i in range(n_circles):
            final_vars[3*i] = best_centers[i, 0]
            final_vars[3*i+1] = best_centers[i, 1]
            final_vars[3*i+2] = best_radii[i]
        
        centers, radii, sum_r = try_optimization(final_vars)
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_centers = centers
            best_radii = radii

    return best_centers, best_radii, best_sum_r

# Note: The problem requires the function to be defined. 
# The code below will run it if executed as a script, but the prompt asks for the function.
# I will leave the execution block commented out or minimal.

if __name__ == "__main__":
    # Just to test locally if needed
    c, r, s = run_packing()
    print(f"Final Sum: {s}")
import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern
    # We aim for a configuration that fits 26 circles.
    # A pattern of rows with lengths 5, 6, 5, 6, 4 sums to 26.
    # Or 6, 5, 6, 5, 4.
    # Let's try to construct a grid first and then perturb.
    
    centers = np.zeros((n, 2))
    
    # Heuristic initialization: Hexagonal packing
    # Rows: 6, 5, 6, 5, 4 (Total 26)
    # We will scale and shift these to fit in [0,1]x[0,1]
    
    row_counts = [6, 5, 6, 5, 4]
    idx = 0
    centers_list = []
    
    # Approximate spacing for hex packing
    # Let's guess a radius r=0.1 initially to layout
    r_init = 0.1
    d = 2 * r_init
    h = np.sqrt(3) * r_init
    
    current_y = r_init
    
    for k, count in enumerate(row_counts):
        # Offset for hexagonal rows: alternate between 0 and d/2 (or r)
        # Actually, standard hex offset is r.
        # But we need to fit in width 1.
        # If row has 'count' circles, width occupied is (count-1)*d + 2r = count*d.
        # If count=6, width 6d = 1.2 > 1. So we can't fit 6 circles of r=0.1 in a row horizontally aligned.
        # However, hexagonal packing allows staggering.
        # But the width constraint for the row itself is determined by the span of centers + 2r.
        # Span of centers for 'count' circles is (count-1)*d.
        # Total width = (count-1)*d + 2r = (count-1)*2r + 2r = 2*count*r = count*d.
        # So a row of 6 circles requires width 6d.
        # To fit in 1, d must be <= 1/6 approx 0.166, so r <= 0.0833.
        # But we want r ~ 0.1.
        # This suggests we cannot have a full row of 6 circles aligned horizontally if r=0.1.
        # BUT, we can rotate the lattice or use a different arrangement.
        # Or, the rows are not perfectly aligned to axes?
        
        # Let's try a square grid initialization first, it's safer for fitting.
        # 5x5 grid fits r=0.1. We have 1 extra circle.
        # We can try to squeeze.
        pass

    # Better Initialization: Randomized or Grid-based perturbation
    # Let's use a 6x6 grid subset or just random points and let optimizer work.
    # But optimizer needs good start.
    
    # Let's try a dense packing generator
    # Place points on a triangular lattice
    # Triangular lattice points: (i*dx + (j%2)*dx/2, j*dy)
    # dx = 2r, dy = sqrt(3)r
    
    # Let's try to find a good r and positions numerically.
    
    # Variables: x, y for 26 circles, and r.
    # But r is common for equal circles optimization.
    
    # Let's define the optimization problem for equal radii first.
    # Variables: [x1, y1, ..., x26, y26, r]
    # Objective: Maximize r -> Minimize -r
    
    # Initial guess: Grid
    # 26 points in [0,1]
    # Use a pseudo-random shuffle of a grid or just linspace
    # A 6x5 grid has 30 points. Remove 4.
    
    grid_x = np.linspace(0.1, 0.9, 6) # 6 points
    grid_y = np.linspace(0.1, 0.9, 5) # 5 points
    # This creates a 6x5 grid. 30 points.
    # We need 26. Remove 4 corners or random ones.
    
    all_x, all_y = np.meshgrid(grid_x, grid_y)
    all_x = all_x.flatten()
    all_y = all_y.flatten()
    
    # Remove 4 points from the ends to get 26
    # Keep indices 0 to 25
    init_x = all_x[:26]
    init_y = all_y[:26]
    
    centers_init = np.column_stack((init_x, init_y))
    r_init = 0.08 # Safe initial radius
    
    # 2. Optimization for Equal Radii
    # We want to maximize r such that circles don't overlap and stay in square.
    
    def objective(vars):
        # vars: [x1, y1, ..., x26, y26, r]
        r = vars[-1]
        return -r # Minimize -r is Maximize r

    def constraint_overlap(vars):
        # vars: [x1, y1, ..., x26, y26, r]
        centers = vars[:-1].reshape(-1, 2)
        r = vars[-1]
        # Check all pairs
        # dist >= 2r
        # dist^2 >= 4r^2
        dists_sq = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dists_sq.append(np.sum(diff**2))
        return np.array(dists_sq) - 4 * r**2

    def constraint_boundary(vars):
        # vars: [x1, y1, ..., x26, y26, r]
        centers = vars[:-1].reshape(-1, 2)
        r = vars[-1]
        # x >= r, x <= 1-r, y >= r, y <= 1-r
        c_list = []
        for i in range(n):
            c_list.append(centers[i, 0] - r)
            c_list.append(1 - centers[i, 0] - r)
            c_list.append(centers[i, 1] - r)
            c_list.append(1 - centers[i, 1] - r)
        return np.array(c_list)

    # Initial guess vector
    x0 = np.concatenate([centers_init.flatten(), [r_init]])
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(n * 2):
        bounds.append((0.0, 1.0))
    bounds.append((0.0, 0.5))

    # Constraints
    # Overlap constraints: g(x) >= 0
    cons_overlap = {'type': 'ineq', 'fun': constraint_overlap}
    cons_boundary = {'type': 'ineq', 'fun': constraint_boundary}
    
    # Solve
    # SLSQP is good for non-linear constraints
    result = opt.minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=[cons_overlap, cons_boundary],
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if result.success:
        opt_centers = result.x[:-1].reshape(-1, 2)
        opt_r = result.x[-1]
    else:
        # Fallback to initial if optimization fails
        opt_centers = centers_init
        opt_r = r_init

    # 3. Refine with unequal radii (Optional but potentially beneficial)
    # Now that we have good centers, we can try to maximize sum of radii.
    # However, the equal radius solution is likely very good.
    # Let's just stick to the equal radius solution or slightly adjust.
    # Actually, we can try to optimize radii individually now.
    # But let's first check the sum.
    
    radii = np.full(n, opt_r)
    sum_radii = np.sum(radii)
    
    # Let's try a local search to see if we can expand radii further.
    # Since the constraints are tight, maybe we can perturb centers to allow larger radii?
    # The optimizer should have done this.
    
    # Let's double check validity
    # If the optimizer found a valid solution, we are good.
    
    # Just to be safe, let's run a few random restarts or perturbations if the result is not great.
    # But SLSQP from a good grid start usually works.
    
    # One issue: SLSQP might get stuck in local optima.
    # Let's try to improve by running the optimization a few times with perturbed starts.
    
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Try a few random perturbations of the grid start
    for trial in range(5):
        # Perturb initial positions slightly
        perturbation = np.random.uniform(-0.01, 0.01, size=(n, 2))
        curr_centers = np.clip(centers_init + perturbation, 0.01, 0.99)
        curr_r = 0.08
        
        x0_curr = np.concatenate([curr_centers.flatten(), [curr_r]])
        
        try:
            res = opt.minimize(
                objective, 
                x0_curr, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[cons_overlap, cons_boundary],
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.success and -res.fun > best_sum / n: # Compare radius
                best_sum = -res.fun * n
                best_centers = res.x[:-1].reshape(-1, 2)
                best_radii = np.full(n, res.x[-1])
            elif -res.fun * n > best_sum:
                 # Even if not strictly better per circle, check total
                 # But for equal radii, total is proportional to r.
                 pass
        except:
            pass
            
        # Also try a different initialization: Hexagonal
        # Generate hex points
        hex_centers = []
        y = 0.1
        while y < 0.9:
            x = 0.1
            row_shift = 0.05 # offset
            if int((y-0.1)/0.17) % 2 == 1:
                x = 0.15
            
            while x < 0.9:
                if len(hex_centers) < 26:
                    hex_centers.append([x, y])
                x += 0.2
            y += 0.1732 # sqrt(3)/2 * 0.2
            
        if len(hex_centers) >= 26:
            hex_arr = np.array(hex_centers[:26])
            # Scale to fit better?
            # Normalize to [0,1]
            hex_arr = (hex_arr - hex_arr.min(axis=0)) / (hex_arr.max(axis=0) - hex_arr.min(axis=0))
            # Center in square
            hex_arr = hex_arr * 0.9 + 0.05
            
            x0_hex = np.concatenate([hex_arr.flatten(), [0.08]])
            try:
                res_hex = opt.minimize(
                    objective, 
                    x0_hex, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=[cons_overlap, cons_boundary],
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                if res_hex.success:
                    r_val = res_hex.x[-1]
                    if 26 * r_val > best_sum:
                        best_sum = 26 * r_val
                        best_centers = res_hex.x[:-1].reshape(-1, 2)
                        best_radii = np.full(n, r_val)
            except:
                pass

    if best_centers is not None:
        centers = best_centers
        radii = best_radii
        sum_radii = np.sum(radii)
    else:
        # Fallback to the first result
        centers = opt_centers
        radii = np.full(n, opt_r)
        sum_radii = np.sum(radii)
        
    # Final check and correction for numerical errors
    # Ensure strictly inside
    centers = np.clip(centers, np.max(radii) + 1e-9, 1 - np.max(radii) - 1e-9)
    
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(centers)
    # print(radii)
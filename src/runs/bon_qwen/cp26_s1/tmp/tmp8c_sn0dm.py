import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Helper function to compute constraints
    def compute_constraints(vars, n):
        # vars shape: (3*n,)
        # Structure: [x0, y0, r0, x1, y1, r1, ...]
        # Actually easier to reshape
        centers = vars[:2*n].reshape(n, 2)
        radii = vars[2*n:]
        
        constraints = []
        
        # Boundary constraints: x - r >= 0, 1 - (x + r) >= 0, etc.
        # x >= r  => x - r >= 0
        # 1 - x >= r => 1 - x - r >= 0
        # y >= r  => y - r >= 0
        # 1 - y >= r => 1 - y - r >= 0
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            constraints.append(x - r)       # left
            constraints.append(1 - x - r)    # right
            constraints.append(y - r)        # bottom
            constraints.append(1 - y - r)    # top
            
        # Overlap constraints: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                constraints.append(dist_sq - r_sum*r_sum)
                
        return np.array(constraints)

    # Objective function: minimize -sum(radii)
    def objective(vars):
        radii = vars[2*n:]
        return -np.sum(radii)

    def objective_with_penalty(vars, mu):
        # Just for reference, but we use constraints directly in SLSQP
        return objective(vars)

    # Function to run optimization from a specific initial guess
    def solve_from_init(x0, n):
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
        # Constraints for SLSQP
        # type 'ineq' means constraint >= 0
        cons = ({'type': 'ineq', 'fun': lambda v: compute_constraints(v, n)})

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 500, 'ftol': 1e-9})
            if res.success:
                return res.x, -res.fun
            else:
                # Return best found so far even if not successful
                current_val = -objective(res.x)
                # Check validity roughly
                if np.all(compute_constraints(res.x, n) >= -1e-6):
                    return res.x, current_val
                return None, 0
        except Exception:
            return None, 0

    best_vars = None
    best_score = 0.0
    
    # Generate initial guesses
    # Hexagonal grid packing
    # We want to fit 26 circles. 
    # A 5x5 grid has 25. A 6x5 has 30.
    # Let's create a dense grid and pick 26 points.
    
    # Hexagonal lattice generation
    # Spacing dx = sqrt(3)/2 * d? No.
    # For equal circles radius r, centers distance 2r.
    # Vertical distance sqrt(3)*r. Horizontal 2r.
    # Let's just place them in a grid pattern and let optimizer fix it.
    
    # Try a few random seeds for perturbation
    np.random.seed(42)
    
    for seed in range(10): # Run 10 trials
        np.random.seed(seed)
        
        # Create a grid of points
        # 6 rows, 5 columns = 30 points. We take first 26.
        # Or 5 rows, 6 columns?
        # Let's try to distribute them evenly.
        
        # Method: Randomly place in square with small radius
        # Better: Hexagonal arrangement
        rows = 6
        cols = 5
        # This gives 30 spots.
        
        # Lattice parameters
        # If we want to fit in [0,1], let's use a loose grid first
        # x coords: 1/(cols+1) * (1..cols)
        # y coords: 1/(rows+1) * (1..rows)
        # But hexagonal shift: even rows shifted by dx/2
        
        points = []
        # Let's try to generate points in a hexagonal pattern scaled to fit [0,1] roughly
        # We want roughly 5 columns and 5-6 rows.
        
        # Let's use a simple grid for initialization, it's robust
        # 6 rows, 5 cols
        xs = np.linspace(0.15, 0.85, cols) # 5 points
        ys = np.linspace(0.15, 0.85, rows) # 6 points
        
        grid_points = []
        for i, y in enumerate(ys):
            row_xs = xs.copy()
            # Shift every other row for hexagonal effect
            if i % 2 == 1:
                row_xs += (xs[1] - xs[0]) / 2
                # Adjust if out of bounds, but linspace keeps it inside
            for x in row_xs:
                grid_points.append([x, y])
        
        # We have 30 points. Shuffle and take 26.
        np.random.shuffle(grid_points)
        initial_centers = np.array(grid_points[:n])
        
        # Initial radii: small valid radius
        # Max possible radius in unit square is 0.5. 
        # With 26 circles, maybe 0.05 is safe start.
        initial_radii = np.full(n, 0.05)
        
        # Flatten to vector
        x0 = np.concatenate([initial_centers.flatten(), initial_radii])
        
        vars_sol, score = solve_from_init(x0, n)
        
        if vars_sol is not None and score > best_score:
            best_score = score
            best_vars = vars_sol
            
    if best_vars is None:
        # Fallback to random
        initial_centers = np.random.uniform(0.2, 0.8, (n, 2))
        initial_radii = np.full(n, 0.02)
        x0 = np.concatenate([initial_centers.flatten(), initial_radii])
        vars_sol, score = solve_from_init(x0, n)
        if vars_sol is not None:
            best_vars = vars_sol
            best_score = score

    # Extract result
    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:]
    
    # Sort by index to ensure consistent output? Not required but good practice.
    # The order doesn't matter for the set of circles.
    
    # Final validation and cleanup
    # Ensure radii are non-negative (should be due to bounds)
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
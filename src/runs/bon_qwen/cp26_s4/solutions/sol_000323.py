# sol_000323 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 50425c04) state=a6157f55 sum of radii=2.556021 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Strategy:
    1. Initialize centers using a hexagonal lattice pattern.
    2. Optimize centers and radii using scipy.optimize.minimize (SLSQP) to maximize sum of radii
       subject to boundary and non-overlap constraints.
    3. Run multiple restarts to find a good global optimum.
    """
    
    n_circles = 26
    
    # Helper function to compute the loss (negative sum of radii)
    def objective(vars_flat):
        # vars_flat shape: (n_circles * 3) -> [x1, y1, r1, x2, y2, r2, ...]
        centers = vars_flat[:n_circles * 2].reshape(-1, 2)
        radii = vars_flat[n_circles * 2:]
        return -np.sum(radii)

    # Helper function to generate inequality constraints
    # SLSQP expects constraints of form: fun(x) >= 0
    def make_constraints():
        constraints = []
        
        # 1. Boundary constraints
        # x - r >= 0
        # x + r <= 1  =>  1 - x - r >= 0
        # y - r >= 0
        # y + r <= 1  =>  1 - y - r >= 0
        
        for i in range(n_circles):
            idx_x = i * 3
            idx_y = i * 3 + 1
            idx_r = n_circles * 3 + i # Wait, vars structure is [all x, all y, all r] in my current thought?
            # Let's stick to the structure used in objective: [x1, y1, x2, y2, ..., r1, r2, ...]
            # No, in objective I did: centers = vars[:2n].reshape(-1, 2), radii = vars[2n:]
            # So indices:
            # x_i is at i * 2
            # y_i is at i * 2 + 1
            # r_i is at 2*n_circles + i
            
            # x_i - r_i >= 0
            c1 = {
                'type': 'ineq',
                'fun': lambda v, i=i: v[i * 2] - v[2 * n_circles + i]
            }
            # 1 - x_i - r_i >= 0
            c2 = {
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[i * 2] - v[2 * n_circles + i]
            }
            # y_i - r_i >= 0
            c3 = {
                'type': 'ineq',
                'fun': lambda v, i=i: v[i * 2 + 1] - v[2 * n_circles + i]
            }
            # 1 - y_i - r_i >= 0
            c4 = {
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[i * 2 + 1] - v[2 * n_circles + i]
            }
            
            constraints.append(c1)
            constraints.append(c2)
            constraints.append(c3)
            constraints.append(c4)

        # 2. Non-overlap constraints
        # dist(i, j) >= r_i + r_j
        # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                c_overlap = {
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.sqrt((v[i * 2] - v[j * 2])**2 + 
                                                       (v[i * 2 + 1] - v[j * 2 + 1])**2) - 
                                                       (v[2 * n_circles + i] + v[2 * n_circles + j])
                }
                constraints.append(c_overlap)
                
        # 3. Non-negative radii
        for i in range(n_circles):
            c_r = {
                'type': 'ineq',
                'fun': lambda v, i=i: v[2 * n_circles + i]
            }
            constraints.append(c_r)
            
        return constraints

    constraints = make_constraints()
    
    # Bounds for variables
    # x, y in [0, 1]
    # r >= 0 (handled by constraint, but bounds help)
    # r <= 1
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for _ in range(n_circles):
        bounds.append((0.0, 0.5)) # r (max possible radius is 0.5)

    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Function to initialize hexagonal packing
    def initialize_hexagonal(n):
        centers = np.zeros((n, 2))
        # Try to fit n circles in a hexagonal grid
        # Estimate rows and cols
        # Area approx n * pi * r^2 ~ 1. r ~ sqrt(1/(n*pi)) ~ 0.11
        # Spacing ~ 2r ~ 0.22
        # Let's try a grid spacing
        
        count = 0
        row = 0
        while count < n:
            # Calculate number of circles in this row
            # Hexagonal rows are shifted by spacing/2
            # Let's define spacing s
            # Vertical distance between rows is s * sqrt(3)/2
            # Horizontal distance is s
            
            # Let's just place them in a dense pattern and let optimizer fix it
            # Simple approach: iterate rows
            # Row height h = 0.2 (approx)
            y = 0.1 + row * 0.15 # Start with some margin
            
            # x spacing
            # If row is even, start at 0.1
            # If row is odd, start at 0.1 + 0.075 (shift)
            
            shift = 0.075 if row % 2 == 1 else 0.0
            x = 0.1 + shift
            
            while x <= 0.9 and count < n:
                centers[count, 0] = x
                centers[count, 1] = y
                count += 1
                x += 0.15
            
            row += 1
            
        return centers

    # We will run optimization a few times with different shuffles of initial positions
    # to escape local minima.
    num_restarts = 5
    
    for k in range(num_restarts):
        # Generate initial centers
        init_centers = initialize_hexagonal(n_circles)
        
        # Shuffle the points slightly or permute to vary order (though order doesn't matter for sum)
        # But permuting might change convergence path.
        np.random.seed(42 + k)
        np.random.shuffle(init_centers)
        
        # Initial radii: small value
        init_radii = np.full(n_circles, 0.05)
        
        # Flatten variables: [x1, y1, ..., xn, yn, r1, ..., rn]
        x0 = np.concatenate([init_centers.flatten(), init_radii])
        
        # Optimize
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < -best_sum_radii:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = res.x[:n_circles * 2].reshape(-1, 2)
                    best_radii = res.x[n_circles * 2:]
        except Exception as e:
            continue

    # If optimization failed to find anything better than trivial, return a valid default
    if best_centers is None:
        # Fallback to a simple grid
        centers = np.zeros((n_circles, 2))
        radii = np.full(n_circles, 0.01)
        # Just place them in a line
        for i in range(n_circles):
            centers[i, 0] = (i + 0.5) / n_circles
            centers[i, 1] = 0.5
        best_sum_radii = np.sum(radii)
        best_centers = centers
        best_radii = radii

    # Final validation check (internal sanity)
    # Though the function returns it, we should ensure constraints are met within tolerance
    # The optimizer should handle this, but numerical noise exists.
    # We clamp radii to be non-negative and centers to be within bounds just in case.
    best_radii = np.maximum(best_radii, 0.0)
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    # Adjust centers if they violate boundary due to radius
    # This is a post-processing step to ensure validity
    for i in range(n_circles):
        r = best_radii[i]
        # Clamp x
        if best_centers[i, 0] - r < 0:
            best_centers[i, 0] = r
        if best_centers[i, 0] + r > 1:
            best_centers[i, 0] = 1 - r
        # Clamp y
        if best_centers[i, 1] - r < 0:
            best_centers[i, 1] = r
        if best_centers[i, 1] + r > 1:
            best_centers[i, 1] = 1 - r
            
    # Re-calculate sum after potential adjustments
    best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii

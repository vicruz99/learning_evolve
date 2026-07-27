import numpy as np
import scipy.optimize as opt
from itertools import combinations

def generate_initial_centers(n, method='hex'):
    """Generates initial centers for n circles in a unit square."""
    centers = np.zeros((n, 2))
    if method == 'hex':
        # Hexagonal packing approximation
        # We need to fit n circles.
        # Estimate radius r to fit n circles roughly.
        # Area argument: n * pi * r^2 * 0.9069 approx 1 => r approx 0.1 for n=26?
        # Let's use a spacing that fits easily, e.g., r=0.05 initially.
        spacing = 0.15 
        row_step = spacing * np.sqrt(3) / 2
        
        count = 0
        row = 0
        while count < n:
            col = 0
            y = 0.05 + row * row_step
            if y > 0.95: # Keep inside
                break
            
            # Offset for odd rows
            x_offset = 0.05 if row % 2 == 0 else 0.05 + spacing/2
            
            while True:
                x = x_offset + col * spacing
                if x > 0.95:
                    break
                if count < n:
                    centers[count] = [x, y]
                    count += 1
                col += 1
            row += 1
    else:
        # Random initialization
        centers = np.random.rand(n, 2) * 0.8 + 0.1 # Keep somewhat central
        
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Best solution storage
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # We will run multiple optimizations
    num_restarts = 10
    
    for restart in range(num_restarts):
        # Generate initial centers
        if restart == 0:
            initial_centers = generate_initial_centers(n, method='hex')
        else:
            # Add noise to best found so far or random
            if best_centers is not None:
                noise = np.random.normal(0, 0.02, size=best_centers.shape)
                current_centers = np.clip(best_centers + noise, 0.05, 0.95)
            else:
                current_centers = generate_initial_centers(n, method='hex')
        
        # Initial radii: small enough to not overlap
        # Estimate distance to nearest neighbor
        dists = np.linalg.norm(current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        initial_radii = np.minimum(min_dists / 2, 0.1) # Cap at 0.1
        
        # Variables: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = current_centers[i, 0]
            x0[3*i+1] = current_centers[i, 1]
            x0[3*i+2] = initial_radii[i]
            
        # Bounds: x, y in [0, 1], r >= 0
        # Actually r <= 0.5 is implicit from x,y bounds, but let's bound r to [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
            
        # Objective: Maximize sum of radii => Minimize negative sum
        def objective(vars):
            radii = vars[2::3]
            return -np.sum(radii)
            
        # Constraints
        def constraint_factory():
            cons = []
            # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
            # x - r >= 0
            # 1 - x - r >= 0
            for i in range(n):
                idx = 3*i
                # x - r >= 0
                def c_bound_x_min(v, i=i, idx=idx):
                    return v[idx] - v[idx+2]
                cons.append({'type': 'ineq', 'fun': c_bound_x_min})
                
                # 1 - x - r >= 0
                def c_bound_x_max(v, i=i, idx=idx):
                    return 1.0 - v[idx] - v[idx+2]
                cons.append({'type': 'ineq', 'fun': c_bound_x_max})
                
                # y - r >= 0
                def c_bound_y_min(v, i=i, idx=idx):
                    return v[idx+1] - v[idx+2]
                cons.append({'type': 'ineq', 'fun': c_bound_y_min})
                
                # 1 - y - r >= 0
                def c_bound_y_max(v, i=i, idx=idx):
                    return 1.0 - v[idx+1] - v[idx+2]
                cons.append({'type': 'ineq', 'fun': c_bound_y_max})
                
            # Non-overlap constraints: dist >= r_i + r_j
            # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            # dist_sq - (r_i + r_j)^2 >= 0
            # Using combinations to iterate pairs
            for i, j in combinations(range(n), 2):
                idx_i = 3*i
                idx_j = 3*j
                
                def c_overlap(v, ii=i, jj=j, idx_i=idx_i, idx_j=idx_j):
                    dx = v[idx_i] - v[idx_j]
                    dy = v[idx_i+1] - v[idx_j+1]
                    dist_sq = dx*dx + dy*dy
                    r_sum = v[idx_i+2] + v[idx_j+2]
                    return dist_sq - r_sum*r_sum
                
                cons.append({'type': 'ineq', 'fun': c_overlap})
            
            return cons

        # Define constraints once
        constraints = constraint_factory()
        
        try:
            # Optimization
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                               options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success or res.nit > 100: # If it ran significantly
                final_vars = res.x
                final_centers = np.zeros((n, 2))
                final_radii = np.zeros(n)
                for i in range(n):
                    final_centers[i, 0] = final_vars[3*i]
                    final_centers[i, 1] = final_vars[3*i+1]
                    final_radii[i] = final_vars[3*i+2]
                
                # Validate and check sum
                # Simple validation check
                valid = True
                # Check boundaries
                for i in range(n):
                    x, y, r = final_centers[i,0], final_centers[i,1], final_radii[i]
                    if r < -1e-5: valid = False
                    if x - r < -1e-9 or x + r > 1 + 1e-9: valid = False
                    if y - r < -1e-9 or y + r > 1 + 1e-9: valid = False
                
                if valid:
                    current_sum = np.sum(final_radii)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = final_centers.copy()
                        best_radii = final_radii.copy()
                        
        except Exception as e:
            # If optimization fails, continue to next restart
            pass

    # Fallback if best is still None or invalid
    if best_centers is None:
        # Generate a safe default packing (small circles)
        best_centers = generate_initial_centers(n, method='hex')
        best_radii = np.full(n, 0.05)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum

# Note: The problem statement asks to return the function definition.
# The function run_packing will be called externally.
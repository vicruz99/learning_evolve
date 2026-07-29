# sol_000294 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fc92aa36) state=f133dd1b sum of radii=2.569946 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square.
    Returns centers, radii, and the sum of radii.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    best_valid = False

    # 1. Generate an initial hexagonal grid layout
    # This provides a dense starting point for the optimizer
    init_centers = []
    init_radii = []
    
    # Parameters for the hex grid
    r_est = 0.095 
    spacing = 2 * r_est
    row_height = spacing * np.sqrt(3) / 2
    
    # We need to fit 26 circles. A pattern of rows: 5, 6, 5, 6, 4 sums to 26.
    # Or a simple grid of roughly 5x6.
    # Let's try to fit a hex grid that approximates a 5x6 density.
    
    row_counts = [6, 5, 6, 5, 4]
    total = sum(row_counts)
    if total < n:
        # Pad if needed (should not happen with these counts)
        row_counts[-1] += (n - total)
        
    current_y = r_est
    row_idx = 0
    
    for k, count in enumerate(row_counts):
        # Check if next row fits
        if current_y + row_height + r_est > 1.0 and row_idx > 0:
            break
            
        # Centering the row
        # If count is 6, width is 11*r. If 5, 9*r.
        # To fit in [0,1], we need to shift.
        width = (count * 2 - 1) * r_est
        shift_x = (1.0 - width) / 2.0
        
        for j in range(count):
            if len(init_centers) < n:
                x = shift_x + j * spacing
                y = current_y
                init_centers.append([x, y])
                init_radii.append(r_est)
        
        current_y += row_height
        row_idx += 1
        
    # Fill remaining circles if any (randomly in valid spots)
    while len(init_centers) < n:
        # Find a spot
        # Just append a small circle in a gap for now
        # Or simply duplicate and let optimizer fix
        init_centers.append([0.5, 0.5])
        init_radii.append(0.01)

    # Convert to numpy arrays
    X0 = np.zeros(3 * n)
    for i in range(n):
        X0[3*i] = init_centers[i][0]
        X0[3*i+1] = init_centers[i][1]
        X0[3*i+2] = init_radii[i]

    # 2. Optimization Function
    def objective(x):
        # Maximize sum of radii -> Minimize negative sum
        radii = x[2::3]
        return -np.sum(radii)

    def constraints_overlap(x):
        """Distance between centers >= sum of radii"""
        cons = []
        centers = np.array([x[i::3] for i in range(3)]) # shape (2, n)
        radii = x[2::3]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((centers[0, i] - centers[0, j])**2 + 
                               (centers[1, i] - centers[1, j])**2)
                cons.append(dist - radii[i] - radii[j])
        return cons

    def constraints_boundary(x):
        """Circles inside [0,1]x[0,1]"""
        cons = []
        for i in range(n):
            x_i = x[3*i]
            y_i = x[3*i+1]
            r_i = x[3*i+2]
            cons.append(x_i - r_i)           # x >= r
            cons.append(1.0 - x_i - r_i)     # x <= 1-r
            cons.append(y_i - r_i)           # y >= r
            cons.append(1.0 - y_i - r_i)     # y <= 1-r
        return cons

    # We will use SLSQP. It requires constraints in the form func(x) >= 0.
    # scipy accepts 'ineq' constraints as func(x) >= 0.
    
    # Combine constraints
    def all_constraints(x):
        c_ov = constraints_overlap(x)
        c_bd = constraints_boundary(x)
        return c_ov + c_bd

    # Optimization run
    # We run multiple times with slight perturbations to find better local optima
    for trial in range(15):
        # Perturb X0
        current_x = X0.copy()
        if trial > 0:
            noise = np.random.normal(0, 0.01, size=current_x.shape)
            current_x += noise
        
        # Ensure positive radii in initial guess
        current_x[2::3] = np.maximum(current_x[2::3], 0.001)
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r

        try:
            # Run optimization
            res = opt.minimize(objective, current_x, method='SLSQP', 
                               bounds=bounds, 
                               constraints={'type': 'ineq', 'fun': all_constraints},
                               options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or (np.all(all_constraints(res.x) >= -1e-7)):
                # Check validity strictly
                centers_opt = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
                radii_opt = res.x[2::3]
                
                # Quick validation check
                valid = True
                for i in range(n):
                    if radii_opt[i] < 0:
                        valid = False; break
                    if radii_opt[i] > centers_opt[i,0] or radii_opt[i] > 1 - centers_opt[i,0] or \
                       radii_opt[i] > centers_opt[i,1] or radii_opt[i] > 1 - centers_opt[i,1]:
                        valid = False; break
                    for j in range(i+1, n):
                        d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                        if d < radii_opt[i] + radii_opt[j] - 1e-9:
                            valid = False; break
                
                if valid:
                    s = np.sum(radii_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers_opt.copy()
                        best_radii = radii_opt.copy()
                        best_valid = True

        except Exception:
            continue

    # If no valid solution found (unlikely), return a safe default (grid)
    if not best_valid:
        # Fallback: 5x5 grid + 1 small
        best_centers = []
        best_radii = []
        for i in range(5):
            for j in range(5):
                best_centers.append([0.1 + 0.2*i, 0.1 + 0.2*j])
                best_radii.append(0.1)
        best_centers.append([0.2, 0.2])
        best_radii.append(0.04)
        best_centers = np.array(best_centers)
        best_radii = np.array(best_radii)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum

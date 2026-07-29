# sol_000169 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 403fd447) state=6b1301c1 sum of radii=2.617024 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n_circles = 26
    
    # Helper to generate initial centers in a hexagonal grid
    def get_initial_centers(n, r_spacing):
        centers = []
        y = r_spacing
        row_idx = 0
        # Generate points in a hexagonal lattice
        while y + r_spacing <= 1.0 + 1e-9:
            offset = r_spacing if row_idx % 2 == 1 else 0
            x = r_spacing + offset
            while x + r_spacing <= 1.0 + 1e-9:
                centers.append([x, y])
                x += 2 * r_spacing
            y += r_spacing * np.sqrt(3)
            row_idx += 1
        
        centers = np.array(centers)
        if len(centers) >= n:
            # Select n points. Using random choice with seed for reproducibility.
            # A simple random subset works well to break symmetry.
            np.random.seed(int(r_spacing * 1000))
            indices = np.random.choice(len(centers), n, replace=False)
            return centers[indices]
        else:
            # If not enough points (unlikely with small spacing), pad with random points
            while len(centers) < n:
                centers = np.vstack([centers, [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]])
            return centers[:n]

    # Objective: Maximize sum of radii => Minimize negative sum
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)

    # Constraint 1: Boundary (circles inside [0,1]x[0,1])
    # x >= r, x <= 1-r, y >= r, y <= 1-r
    def boundary_constraint(vars):
        con = []
        for i in range(n_circles):
            idx = 3 * i
            x = vars[idx]
            y = vars[idx+1]
            r = vars[idx+2]
            con.append(x - r)
            con.append(1 - x - r)
            con.append(y - r)
            con.append(1 - y - r)
        return np.array(con)

    # Constraint 2: Non-overlap (distance >= sum of radii)
    def overlap_constraints(vars):
        con = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                idx_i = 3 * i
                idx_j = 3 * j
                x_i, y_i, r_i = vars[idx_i], vars[idx_i+1], vars[idx_i+2]
                x_j, y_j, r_j = vars[idx_j], vars[idx_j+1], vars[idx_j+2]
                
                dx = x_i - x_j
                dy = y_i - y_j
                dist = np.hypot(dx, dy)
                con.append(dist - r_i - r_j)
        return np.array(con)

    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n_circles

    best_sum = -np.inf
    best_vars = None

    # Try optimization from multiple initial configurations (hexagonal grids with different densities)
    spacings = [0.04, 0.05, 0.06, 0.07, 0.08]
    # Corresponding safe initial radii to ensure no overlaps at start
    initial_radii_map = {0.04: 0.015, 0.05: 0.02, 0.06: 0.025, 0.07: 0.03, 0.08: 0.035}
    
    for sp in spacings:
        centers_init = get_initial_centers(n_circles, sp)
        r_init = initial_radii_map.get(sp, 0.02)
        
        x0 = np.zeros(3 * n_circles)
        for i in range(n_circles):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = r_init
        
        cons = [
            {'type': 'ineq', 'fun': boundary_constraint},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        try:
            # Use SLSQP solver for constrained optimization
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 200, 'ftol': 1e-7, 'disp': False})
            
            if not np.isnan(res.fun):
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_vars = res.x
        except Exception:
            pass

    # Fallback to a valid grid packing if optimization fails
    if best_vars is None:
        best_vars = np.zeros(3 * n_circles)
        idx = 0
        # 5x5 grid of radius 0.1
        for r in range(5):
            for c in range(5):
                if idx < n_circles:
                    best_vars[3*idx] = 0.1 + 0.2*c
                    best_vars[3*idx+1] = 0.1 + 0.2*r
                    best_vars[3*idx+2] = 0.1
                    idx += 1
        while idx < n_circles:
            best_vars[3*idx] = 0.5
            best_vars[3*idx+1] = 0.5
            best_vars[3*idx+2] = 0.0
            idx += 1

    # Extract results
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    for i in range(n_circles):
        centers[i, 0] = best_vars[3*i]
        centers[i, 1] = best_vars[3*i+1]
        radii[i] = best_vars[3*i+2]
    
    # Post-processing to strictly enforce constraints (handling numerical errors)
    radii = np.maximum(radii, 0)
    
    # Ensure circles are within bounds
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        min_dist = min(x, 1-x, y, 1-y)
        if min_dist < r:
            radii[i] = min_dist
    
    # Resolve any remaining overlaps by shrinking radii
    for _ in range(100):
        overlap_found = False
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = np.hypot(dx, dy)
                sum_r = radii[i] + radii[j]
                # Allow tiny tolerance
                if dist < sum_r - 1e-12:
                    overlap = sum_r - dist
                    reduction = overlap / 2 + 1e-6
                    radii[i] -= reduction
                    radii[j] -= reduction
                    overlap_found = True
        if not overlap_found:
            break
            
    radii = np.maximum(radii, 0)
    
    return centers, radii, np.sum(radii)

if __name__ == "__main__":
    # Execute and print results for verification
    centers, radii, total_sum = run_packing()
    # print(f"Total Sum of Radii: {total_sum}")
    # print("Valid:", validate_packing(centers, radii))

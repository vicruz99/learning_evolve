import numpy as np
import math
from scipy.optimize import minimize

def get_hexagonal_centers(n_circles, r_init):
    """
    Generates initial centers for n_circles in a hexagonal pattern
    that fits within the unit square for a given radius r_init.
    """
    centers = []
    # Hexagonal packing parameters
    # Horizontal spacing: 2r
    # Vertical spacing: sqrt(3)r
    # Row offset: r
    
    # We estimate how many rows and cols we can fit
    # To be safe, we generate a grid slightly larger than needed and pick first n
    # Or just fill row by row.
    
    # Let's try to fill row by row
    row_idx = 0
    y = r_init
    while len(centers) < n_circles:
        # Determine x start for this row
        # Even rows (0, 2, ...): start at r
        # Odd rows (1, 3, ...): start at 2r (shifted by r relative to even row spacing? 
        # Actually, standard hex: row 0 centers at r, 3r, 5r... 
        # row 1 centers at 2r, 4r, 6r...
        # This implies horizontal spacing is 2r.
        
        x_start = r_init if row_idx % 2 == 0 else 2 * r_init
        x = x_start
        
        while x <= 1.0 - r_init:
            centers.append([x, y])
            x += 2 * r_init
            if len(centers) >= n_circles:
                break
        
        y += math.sqrt(3) * r_init
        row_idx += 1
        
        # Safety break to prevent infinite loop if r_init is too large for geometry
        if y > 2.0:
            break
            
    return np.array(centers[:n_circles])

def run_packing():
    n = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None
    
    # Initial radius guess for grid generation
    # A value small enough to fit many circles, e.g., 0.05
    r_grid = 0.05
    
    # Number of restarts to find a better local optimum
    n_restarts = 10
    
    for restart in range(n_restarts):
        # 1. Generate initial centers
        # We use a slightly randomized offset to the grid to avoid symmetry traps
        centers_init = get_hexagonal_centers(n, r_grid)
        
        # Add small random perturbation
        jitter = np.random.uniform(-0.005, 0.005, centers_init.shape)
        centers_init = centers_init + jitter
        
        # Clip to ensure valid initial bounds (though optimizer handles it, good for sanity)
        centers_init[:, 0] = np.clip(centers_init[:, 0], 0.01, 0.99)
        centers_init[:, 1] = np.clip(centers_init[:, 1], 0.01, 0.99)
        
        # 2. Define variables: [x1, y1, r1, x2, y2, r2, ...]
        # Flatten centers and radii
        x0 = np.hstack([centers_init.flatten(), np.full(n, 0.05)])
        
        # 3. Define objective
        def objective(vars):
            r = vars[2::3]
            return -np.sum(r)
        
        # 4. Define constraints
        constraints = []
        bounds = []
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        for i in range(n):
            bounds.append((0.0, 1.0)) # x_i
            bounds.append((0.0, 1.0)) # y_i
            bounds.append((0.0, 0.5)) # r_i
            
        # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        # Converted to g(x) >= 0
        # x - r >= 0  => x - r >= 0
        # 1 - x - r >= 0
        # y - r >= 0
        # 1 - y - r >= 0
        
        for i in range(n):
            idx_x = 3 * i
            idx_y = 3 * i + 1
            idx_r = 3 * i + 2
            
            # x >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[idx_x] - v[idx_r]
            })
            # 1 - x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]
            })
            # y >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[idx_y] - v[idx_r]
            })
            # 1 - y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]
            })

        # Pairwise non-overlap constraints
        # dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_xi, idx_yi, idx_ri = 3*i, 3*i+1, 3*i+2
                idx_xj, idx_yj, idx_rj = 3*j, 3*j+1, 3*j+2
                
                def dist_con(v, i=i, j=j):
                    xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
                    xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r = ri + rj
                    return dist_sq - sum_r**2
                
                constraints.append({
                    'type': 'ineq',
                    'fun': dist_con
                })

        # 5. Optimize
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = res.x[0::3].reshape(n, 2)
                    best_radii = res.x[2::3]
        except Exception as e:
            # Continue if optimization fails
            pass

    # If no valid solution found (unlikely), return a default grid
    if best_centers is None:
        # Fallback to simple grid
        centers_grid = np.zeros((n, 2))
        radii_grid = np.full(n, 0.0)
        # Just a placeholder, logic above should find something
        k = 0
        for i in range(5):
            for j in range(5):
                if k < n:
                    centers_grid[k] = [0.1 + i*0.2, 0.1 + j*0.2]
                    radii_grid[k] = 0.09
                    k += 1
        # Fill remaining
        while k < n:
            centers_grid[k] = [0.5, 0.5]
            radii_grid[k] = 0.0
            k += 1
        best_centers = centers_grid
        best_radii = radii_grid
        best_sum_radii = np.sum(best_radii)

    # Final Validation and Adjustment
    # Ensure radii are not negative due to numerical issues
    best_radii = np.maximum(best_radii, 0.0)
    
    # Center clamping just in case
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        # Clamp center to be valid for radius
        best_centers[i, 0] = np.clip(x, r, 1.0 - r)
        best_centers[i, 1] = np.clip(y, r, 1.0 - r)

    return best_centers, best_radii, float(best_sum_radii)

if __name__ == "__main__":
    # We need to make sure run_packing is callable and returns the tuple
    # The code above defines it.
    pass
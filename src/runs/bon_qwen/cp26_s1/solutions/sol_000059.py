# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fcf75c21) state=3645fcd9 sum of radii=2.568067 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # --- 1. Initialization (Hexagonal Packing) ---
    # We use a staggered grid to approximate a dense packing.
    # Rows: 5, 4, 5, 4, 5, 3 circles.
    rows_config = [5, 4, 5, 4, 5, 3]
    
    centers_init = []
    # Estimate an initial radius to ensure valid starting positions
    r_init = 0.08 
    
    y_curr = r_init
    for i, count in enumerate(rows_config):
        # Stagger odd rows (index 0, 2, 4) by shifting x
        shift = r_init if (i % 2 == 0) else 0.0
        
        # Calculate starting x to center the row
        # Total width of 'count' circles is (count-1)*2r + 2r = count*2r
        # Start at 0.5 - (count-1)*r
        x_start = 0.5 - (count - 1) * r_init + shift
        
        for j in range(count):
            x = x_start + j * 2 * r_init
            centers_init.append([x, y_curr])
        
        # Move to next row (hexagonal vertical spacing)
        y_curr += r_init * np.sqrt(3)

    centers_init = np.array(centers_init)
    # Ensure we have exactly 26
    assert len(centers_init) == n, f"Expected 26 circles, got {len(centers_init)}"

    # --- 2. Optimization Setup ---
    
    def objective(vars):
        # vars contains [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        # We want to maximize sum(r), so minimize -sum(r)
        radii = vars[2::3]
        return -np.sum(radii)

    def boundary_constraints(vars):
        cons = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i + 1]
            r = vars[3*i + 2]
            
            # x - r >= 0
            cons.append(x - r)
            # 1 - (x + r) >= 0  => x + r <= 1
            cons.append(1 - (x + r))
            # y - r >= 0
            cons.append(y - r)
            # 1 - (y + r) >= 0 => y + r <= 1
            cons.append(1 - (y + r))
        return cons

    def non_overlap_constraints(vars):
        cons = []
        for i in range(n):
            xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                
                # dist >= sum_r => dist^2 >= sum_r^2
                # To avoid sqrt, we can check dist_sq - sum_r^2 >= 0
                # However, for differentiability and optimization stability,
                # sometimes dist - sum_r >= 0 is better, but let's use squared.
                # Note: This is non-convex but works with local solvers.
                cons.append(dist_sq - sum_r**2)
        return cons

    # --- 3. Run Optimization ---
    
    # Initial vector
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = r_init

    # Bounds: 0 <= x, y <= 1; 0 <= r <= 0.5
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])

    constraints_list = []
    
    # Add boundary constraints
    # Note: scipy.optimize expects constraints to be returned as a list of dicts or a single dict.
    # For multiple inequalities, we can pass a list of dicts or one dict with a vector function.
    # We will use a list of dicts for clarity, though it's slower.
    # Better: Vectorized constraint function.
    
    def vec_boundary(vars):
        out = np.zeros(4 * n)
        for i in range(n):
            idx = 4 * i
            x = vars[3*i]
            y = vars[3*i + 1]
            r = vars[3*i + 2]
            out[idx] = x - r
            out[idx+1] = 1 - (x + r)
            out[idx+2] = y - r
            out[idx+3] = 1 - (y + r)
        return out

    def vec_overlap(vars):
        m = n * (n - 1) // 2
        out = np.zeros(m)
        k = 0
        for i in range(n):
            xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                out[k] = (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                k += 1
        return out

    constr_bound = {'type': 'ineq', 'fun': vec_boundary}
    constr_overlap = {'type': 'ineq', 'fun': vec_overlap}
    
    # Optimization
    # Using SLSQP which handles bounds and constraints
    try:
        res = scipy.optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[constr_bound, constr_overlap],
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )
        
        best_vars = res.x
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial if optimization fails
        best_vars = x0

    # --- 4. Extract Results ---
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = best_vars[3*i]
        final_centers[i, 1] = best_vars[3*i+1]
        final_radii[i] = best_vars[3*i+2]
        
        # Ensure radii are non-negative (numerical safety)
        if final_radii[i] < 0:
            final_radii[i] = 0.0
            # If radius is 0, position doesn't matter much, but keep it valid
            final_centers[i, 0] = 0.5
            final_centers[i, 1] = 0.5

    total_sum = np.sum(final_radii)
    
    return final_centers, final_radii, total_sum

# Validation check (optional, for local testing)
if __name__ == "__main__":
    import numpy as np
    centers, radii, s = run_packing()
    
    # Run the provided validation function
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any(): return False
        if np.isnan(radii).any(): return False
        for i in range(n):
            if radii[i] < 0: return False
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12: return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12: return False
        return True

    is_valid = validate_packing(centers, radii)
    print(f"Valid: {is_valid}, Sum of Radii: {s:.5f}")

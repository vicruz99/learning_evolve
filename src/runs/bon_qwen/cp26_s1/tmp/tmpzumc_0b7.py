import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Define the objective function: maximize sum of radii
    # We minimize the negative sum
    def objective(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ...]
        # Shape is (3*n,)
        radii = vars[2::3]
        return -np.sum(radii)

    # Define constraints
    # 1. Boundary constraints: r_i <= x_i <= 1 - r_i, r_i <= y_i <= 1 - r_i
    #    x_i - r_i >= 0  => r_i - x_i <= 0
    #    x_i + r_i <= 1  => x_i + r_i - 1 <= 0
    #    y_i - r_i >= 0  => r_i - y_i <= 0
    #    y_i + r_i <= 1  => y_i + r_i - 1 <= 0
    
    # 2. Non-overlap constraints: dist(i, j) >= r_i + r_j
    #    sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    #    (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    #    (ri+rj)^2 - dist^2 <= 0

    def constraint_boundary(vars):
        cons = []
        for i in range(n):
            idx = 3 * i
            xi = vars[idx]
            yi = vars[idx+1]
            ri = vars[idx+2]
            
            # x - r >= 0
            cons.append(xi - ri)
            # 1 - x - r >= 0 => x + r - 1 <= 0 is not standard form for fun >= 0
            # scipy constraints are fun(x) >= 0
            # So we need 1 - x - r >= 0
            cons.append(1.0 - xi - ri)
            # y - r >= 0
            cons.append(yi - ri)
            # 1 - y - r >= 0
            cons.append(1.0 - yi - ri)
            
            # r >= 0 (handled by bounds usually, but can add here)
            cons.append(ri)
            
        return np.array(cons)

    def constraint_non_overlap(vars):
        cons = []
        for i in range(n):
            idx_i = 3 * i
            xi = vars[idx_i]
            yi = vars[idx_i+1]
            ri = vars[idx_i+2]
            
            for j in range(i + 1, n):
                idx_j = 3 * j
                xj = vars[idx_j]
                yj = vars[idx_j+1]
                rj = vars[idx_j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                sum_r = ri + rj
                
                # dist >= sum_r  <=> dist^2 >= sum_r^2  <=> dist^2 - sum_r^2 >= 0
                # We need fun(x) >= 0
                cons.append(dist_sq - sum_r*sum_r)
                
        return np.array(cons)

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square)
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-5, 0.5)) # r (lower bound > 0 to avoid degenerate solutions)

    # Initial guesses
    best_result = None
    best_sum = -np.inf
    
    # Try multiple random starts
    for _ in range(10):
        # Generate random initial positions
        rng = np.random.default_rng(np.random.randint(0, 10000))
        
        x_init = rng.random(n)
        y_init = rng.random(n)
        
        # Initial radii: small to ensure feasibility
        r_init = np.full(n, 0.05)
        
        # Construct initial vector
        init_vars = np.zeros(3 * n)
        for i in range(n):
            init_vars[3*i] = x_init[i]
            init_vars[3*i+1] = y_init[i]
            init_vars[3*i+2] = r_init[i]
            
        # Constraints setup
        # Boundary constraints
        constr_b = {'type': 'ineq', 'fun': constraint_boundary}
        # Non-overlap constraints
        constr_n = {'type': 'ineq', 'fun': constraint_non_overlap}
        
        try:
            res = opt.minimize(objective, init_vars, method='SLSQP', 
                              bounds=bounds, 
                              constraints=[constr_b, constr_n],
                              options={'ftol': 1e-12, 'maxiter': 1000})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res.x
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue

    if best_result is None:
        # Fallback to a simple grid if optimization fails
        centers = np.array([[0.1, 0.1]])
        # Just create a dummy output to avoid crash, though this should not happen
        raise RuntimeError("Optimization failed to find any solution")

    # Extract centers and radii
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_result[3*i]
        centers[i, 1] = best_result[3*i+1]
        radii[i] = best_result[3*i+2]
        
    return centers, radii, np.sum(radii)

if __name__ == "__main__":
    # To test locally
    import math
    def validate_packing_local(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any() or np.isnan(radii).any():
            return False
        for i in range(n):
            if radii[i] < -1e-12: return False
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True

    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing_local(c, r)}")
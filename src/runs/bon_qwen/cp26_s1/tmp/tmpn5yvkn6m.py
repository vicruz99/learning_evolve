import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # We use a penalty method to maximize sum of radii subject to constraints.
    # Variables: [r1, u1, v1, r2, u2, v2, ..., rn, un, vn]
    # where center (x, y) = (r + u, r + v).
    # This transformation ensures x >= r and y >= r (since u, v >= 0).
    # Boundary constraints x + r <= 1 and y + r <= 1 become 2r + u <= 1 and 2r + v <= 1.
    
    penalty_weight = 10000.0
    
    def cost_function(vars):
        r = vars[0::3]
        u = vars[1::3]
        v = vars[2::3]
        
        # Objective: Minimize -sum(r)
        obj = -np.sum(r)
        
        penalty = 0.0
        
        # Boundary penalties: 2*r + u <= 1 and 2*r + v <= 1
        # Violation: max(0, 2*r + u - 1)
        viol_u = np.maximum(0, 2.0 * r + u - 1.0)
        viol_v = np.maximum(0, 2.0 * r + v - 1.0)
        
        penalty += np.sum(viol_u**2) + np.sum(viol_v**2)
        
        # Overlap penalties: dist(i, j) >= r_i + r_j
        x = r + u
        y = r + v
        
        # Compute pairwise distances and radius sums
        X_diff = x[:, np.newaxis] - x[np.newaxis, :]
        Y_diff = y[:, np.newaxis] - y[np.newaxis, :]
        D = np.sqrt(X_diff**2 + Y_diff**2)
        
        R_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Mask for upper triangle (i < j)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Violation: max(0, R_sum - D)
        diff = R_sum - D
        diff[~mask] = 0.0
        viol_overlap = np.maximum(0, diff)
        
        penalty += np.sum(viol_overlap**2)
        
        return obj + penalty_weight * penalty

    # Bounds: r, u, v >= 0. 
    # r <= 0.5, u <= 1, v <= 1 (loose upper bounds)
    bounds = [(0, 1) for _ in range(3 * n)]
    
    best_score = -float('inf')
    best_centers = None
    best_radii = None
    
    # Multi-start optimization
    num_restarts = 100
    
    for seed in range(num_restarts):
        np.random.seed(seed)
        
        # Initialize with small radii to ensure initial feasibility
        r_init = np.random.uniform(0.01, 0.03, n)
        max_u = 1.0 - 2.0 * r_init
        max_v = 1.0 - 2.0 * r_init
        u_init = np.random.uniform(0, max_u)
        v_init = np.random.uniform(0, max_v)
        
        vars_init = np.empty(3 * n)
        vars_init[0::3] = r_init
        vars_init[1::3] = u_init
        vars_init[2::3] = v_init
        
        try:
            res = scipy.optimize.minimize(
                cost_function, 
                vars_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            
            # Extract solution
            r_sol = res.x[0::3]
            u_sol = res.x[1::3]
            v_sol = res.x[2::3]
            
            # Strict validation
            if np.any(r_sol < 0) or np.any(u_sol < 0) or np.any(v_sol < 0):
                continue
            if np.any(2.0 * r_sol + u_sol > 1.0 + 1e-9) or np.any(2.0 * r_sol + v_sol > 1.0 + 1e-9):
                continue
            
            x_sol = r_sol + u_sol
            y_sol = r_sol + v_sol
            centers_sol = np.column_stack((x_sol, y_sol))
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((centers_sol[i, 0] - centers_sol[j, 0])**2 + 
                                   (centers_sol[i, 1] - centers_sol[j, 1])**2)
                    if dist < r_sol[i] + r_sol[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = np.sum(r_sol)
                if score > best_score:
                    best_score = score
                    best_centers = centers_sol.copy()
                    best_radii = r_sol.copy()
                    
        except Exception:
            continue

    if best_centers is not None:
        return best_centers, best_radii, best_score
    
    return generate_fallback()

def generate_fallback() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    radii = np.zeros(n)
    centers = np.zeros((n, 2))
    
    # 5x5 grid of radius 0.1
    r_grid = 0.1
    coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    idx = 0
    for y in coords:
        for x in coords:
            centers[idx] = [x, y]
            radii[idx] = r_grid
            idx += 1
            
    # 26th circle in a gap
    centers[25] = [0.2, 0.2]
    radii[25] = 0.041
    
    return centers, radii, np.sum(radii)
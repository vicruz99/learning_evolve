import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars_flat):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_flat[2*N_CIRCLES:])

def constraints(vars_flat):
    """Inequality constraints: boundary containment and non-overlap."""
    c = vars_flat[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r = vars_flat[2*N_CIRCLES:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con = [
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ]
    
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(N_CIRCLES):
        dx = c[i, 0] - c[i+1:, 0]
        dy = c[i, 1] - c[i+1:, 1]
        dist_sq = dx**2 + dy**2
        con.append(dist_sq - (r[i] + r[i+1:])**2)
        
    return np.concatenate(con)

def run_packing():
    np.random.seed(42)
    n = N_CIRCLES
    
    # 1. Hexagonal lattice initialization for high packing density baseline
    points = []
    s = 0.23
    for row in range(8):
        y = 0.12 + row * s * np.sqrt(3)/2
        if y > 0.95: break
        offset = (s/2) if row % 2 == 1 else 0
        for col in range(7):
            x = 0.12 + col * s + offset
            if x > 0.95: break
            points.append((x, y))
            
    initial_centers = np.array(points[:n])
    # Add perturbation to break grid symmetry
    initial_centers += np.random.uniform(-0.01, 0.01, initial_centers.shape)
    initial_centers = np.clip(initial_centers, 0.05, 0.95)
    
    initial_radii = np.full(n, 0.04)
    initial_radii += np.random.uniform(0, 0.005, n)
    
    best_res = None
    best_score = -np.inf
    
    cons = {'type': 'ineq', 'fun': constraints}
    bounds = [(0, 1)]*(2*n) + [(0, 0.5)]*n
    
    # 2. Multi-start optimization to find best local optimum
    for trial in range(5):
        x0 = np.concatenate([initial_centers.flatten(), initial_radii])
        # Perturb initial guess for each trial
        x0 += np.random.uniform(-0.005, 0.005, x0.shape)
        x0[:2*n] = np.clip(x0[:2*n], 0.01, 0.99)
        x0[2*n:] = np.clip(x0[2*n:], 0.01, 0.4)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-11, 'disp': False})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_res = res
        except Exception:
            pass
            
    # 3. Extract and return results
    if best_res is None:
        final_centers = initial_centers
        final_radii = initial_radii
    else:
        final_centers = best_res.x[:2*n].reshape(n, 2)
        final_radii = best_res.x[2*n:]
        
    return final_centers, final_radii, np.sum(final_radii)
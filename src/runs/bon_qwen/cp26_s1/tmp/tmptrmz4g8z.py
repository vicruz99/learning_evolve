import numpy as np
from scipy.optimize import minimize

N = 26

def constraint_fun(v):
    """Compute all inequality constraints for the optimizer."""
    c = []
    # Boundary constraints for each circle
    for i in range(N):
        xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
        c.append(xi - ri)
        c.append(1.0 - xi - ri)
        c.append(yi - ri)
        c.append(1.0 - yi - ri)
        
    # Pairwise non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
            dx = xi - xj
            dy = yi - yj
            c.append(dx*dx + dy*dy - (ri + rj)**2)
            
    return np.array(c)

def objective_fun(v):
    """Return negative sum of radii for minimization."""
    total = 0.0
    for i in range(N):
        total += v[3*i+2]
    return -total

def check_and_fix(centers, radii):
    """Ensure strict feasibility within validator tolerance."""
    # Clamp centers to valid region given current radii
    for i in range(N):
        radii[i] = max(radii[i], 0.0)
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
        
    # Iteratively scale down radii if overlaps persist due to numerical precision
    scale = 1.0
    for _ in range(20):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        scale *= 0.9995
        radii *= scale
        
    return centers, radii

def run_packing():
    np.random.seed(42)
    
    # Hexagonal lattice initialization for better density
    centers_init = np.zeros((N, 2))
    radii_init = np.ones(N) * 0.08
    idx = 0
    spacing_x = 0.22
    spacing_y = spacing_x * np.sqrt(3) / 2
    row = 0
    while idx < N:
        cols = 5 if row % 2 == 0 else 4
        for col in range(cols):
            if idx >= N: break
            x = 0.1 + col * spacing_x + (row % 2) * spacing_x / 2
            y = 0.1 + row * spacing_y
            centers_init[idx] = [x, y]
            idx += 1
        row += 1
        
    x0_base = np.concatenate([centers_init.flatten(), radii_init])
    
    bounds = [(0.0, 1.0) for _ in range(2*N)] + [(1e-8, 0.5) for _ in range(N)]
    cons = [{'type': 'ineq', 'fun': constraint_fun}]
    
    best_sum = -1.0
    best_c = None
    best_r = None
    
    # Multi-start optimization to avoid local minima
    for k in range(4):
        x0 = x0_base.copy()
        x0 += np.random.randn(len(x0)) * 0.005
        
        # Ensure initial guess respects bounds
        for i in range(2*N):
            x0[i] = np.clip(x0[i], 0.0, 1.0)
        for i in range(N):
            x0[2*N + i] = np.clip(x0[2*N + i], 1e-8, 0.5)
            
        res = minimize(objective_fun, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 2000})
                       
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_c = res.x[:2*N].reshape(N, 2)
            best_r = res.x[2*N:].copy()
            
    best_c, best_r = check_and_fix(best_c, best_r)
    final_sum = np.sum(best_r)
    
    return best_c, best_r, final_sum
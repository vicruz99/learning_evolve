import numpy as np
from scipy.optimize import minimize, linprog
import math

def compute_penalty(centers, radii, lam):
    """Computes the objective: negative sum of radii + penalty for violations."""
    n = centers.shape[0]
    energy = -np.sum(radii)
    
    # Inter-circle overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                energy += lam * overlap**2
                
    # Boundary violation penalties
    for i in range(n):
        x, y = centers[i]
        dist_bdry = min(x, 1.0 - x, y, 1.0 - y)
        overlap = radii[i] - dist_bdry
        if overlap > 0:
            energy += lam * overlap**2
            
    return energy

def objective_wrapper(x, n, lam):
    """Wrapper to unpack variables and call the penalty function."""
    centers = x[:2*n].reshape((n, 2))
    radii = x[2*n:]
    return compute_penalty(centers, radii, lam)

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Hexagonal Initialization
    centers = np.zeros((n, 2))
    k = 0
    dx, dy = 0.22, 0.22 * math.sqrt(3) / 2.0
    y = 0.15
    while k < n:
        x = 0.15 if (int((y - 0.15)/dy) % 2 == 0) else 0.15 + dx/2
        while x < 0.9 and k < n:
            centers[k] = [x, y]
            k += 1
            x += dx
        y += dy
    while k < n:
        centers[k] = np.random.uniform(0.1, 0.9, 2)
        k += 1
        
    radii = np.full(n, 0.06)
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x,y in [0,1], r in [1e-4, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)])
        
    current_x = x0.copy()
    
    # 2. Iterative Penalty Optimization
    for epoch in range(15):
        lam = 2000.0 * (2.0 ** epoch)
        res = minimize(objective_wrapper, current_x, args=(n, lam), 
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        current_x = res.x
        
        # Escape local minima
        if epoch % 5 == 4:
            current_x[:2*n] += np.random.normal(0, 0.008, (2*n))
            current_x[:2*n] = np.clip(current_x[:2*n], 0.0, 1.0)
            
    centers_opt = current_x[:2*n].reshape((n, 2))
    
    # 3. Exact Radii Assignment via Linear Programming
    # Maximize sum(r_i) subject to r_i + r_j <= dist_ij and r_i <= dist_bdry_i
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Inter-circle constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(np.linalg.norm(centers_opt[i] - centers_opt[j]))
            
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(min(centers_opt[i][0], 1.0 - centers_opt[i][0],
                        centers_opt[i][1], 1.0 - centers_opt[i][1]))
                        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        radii_opt = res_lp.x
    except Exception:
        # Fallback projection if LP fails
        radii_opt = np.full(n, 0.05)
        for i in range(n):
            d_b = min(centers_opt[i][0], 1.0-centers_opt[i][0], centers_opt[i][1], 1.0-centers_opt[i][1])
            radii_opt[i] = min(radii_opt[i], d_b)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                if radii_opt[i] + radii_opt[j] > d:
                    overlap = radii_opt[i] + radii_opt[j] - d
                    radii_opt[i] -= overlap/2
                    radii_opt[j] -= overlap/2
                    
    total_sum = np.sum(radii_opt)
    return centers_opt, radii_opt, float(total_sum)
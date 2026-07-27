import numpy as np
from scipy.optimize import linprog, minimize

def solve_radii(centers):
    """
    Given fixed centers, solve the LP to find radii that maximize sum of radii
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    c = -np.ones(n)
    
    # Precompute pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= dist_to_boundary
    for i in range(n):
        d_b = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(d_b)
        
        # Pairwise constraints: r_i + r_j <= dist(i, j)
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return -res.fun, res.x
    return 0.0, np.zeros(n)

def objective(x):
    """Objective function for scipy optimizer: minimize negative sum of radii"""
    centers = x.reshape(-1, 2)
    val, _ = solve_radii(centers)
    return -val

def run_packing():
    np.random.seed(42)
    best_val = 0.0
    best_centers = None
    best_radii = None
    
    inits = []
    
    # Initialization 1: Perturbed 5x6 Grid
    x = np.linspace(0.125, 0.875, 5)
    y = np.linspace(1/12, 11/12, 6)
    gx, gy = np.meshgrid(x, y)
    inits.append(np.column_stack([gx.ravel(), gy.ravel()])[:26])
    
    # Initialization 2: Hexagonal Pattern
    hex_centers = []
    row_counts = [6, 5, 6, 5, 4]
    y_pos = 0.1
    dy = 0.22
    for r_idx, count in enumerate(row_counts):
        x_start = 0.5 - (count - 1) * 0.09
        for c_idx in range(count):
            hex_centers.append([x_start + c_idx * 0.18, y_pos])
        y_pos += dy
        if r_idx < len(row_counts) - 1:
            # Shift x for next row to interleave
            pass 
    inits.append(np.array(hex_centers))

    for init in inits:
        for _ in range(6):
            c0 = init + np.random.randn(26, 2) * 0.04
            c0 = np.clip(c0, 0.01, 0.99)
            
            try:
                res = minimize(objective, c0.ravel(), method='Nelder-Mead', 
                               options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-5})
                val, radii = solve_radii(res.x.reshape(-1, 2))
                if val > best_val:
                    best_val = val
                    best_centers = res.x.reshape(-1, 2)
                    best_radii = radii
            except Exception:
                continue
                
    # Ensure exact feasibility
    if best_centers is not None:
        _, best_radii = solve_radii(best_centers)
        # Clip radii slightly to avoid boundary precision issues
        best_radii = np.clip(best_radii, 1e-9, None)
        best_val = np.sum(best_radii)
        
    return best_centers, best_radii, best_val
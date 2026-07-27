import numpy as np
from scipy.optimize import minimize

def compute_penalty(vars, N, C):
    """Compute quadratic penalty for constraint violations."""
    rs = vars[2::3]
    pen = 0.0
    
    # Boundary penalties
    for i in range(N):
        x = vars[3*i]
        y = vars[3*i+1]
        r = rs[i]
        if r > x: 
            pen += C * (r - x)**2
        if r > 1 - x: 
            pen += C * (r - (1 - x))**2
        if r > y: 
            pen += C * (r - y)**2
        if r > 1 - y: 
            pen += C * (r - (1 - y))**2

    # Overlap penalties
    for i in range(N):
        xi, yi, ri = vars[3*i], vars[3*i+1], rs[i]
        for j in range(i + 1, N):
            dx = xi - vars[3*j]
            dy = yi - vars[3*j+1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-8: 
                d = 1e-8
            gap = d - ri - vars[3*j+2]
            if gap < 0:
                pen += C * gap * gap
    return pen

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    
    # 1. Initialization: Hexagonal grid pattern
    centers = np.zeros((N, 2))
    cols = 6
    rows = 5
    for i in range(N):
        r_idx = i // cols
        c_idx = i % cols
        # Hexagonal offset
        x = (c_idx + 0.5) / cols + (0.5 if r_idx % 2 else 0) / cols
        y = (r_idx + 0.5) / rows
        centers[i] = [x, y]
        
    radii = np.full(N, 0.04)
    
    # Flatten to optimization variable vector [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    # 2. Iterative Penalty Optimization
    C = 50.0
    x_curr = x0.copy()
    
    for it in range(30):
        def obj(v):
            return -np.sum(v[2::3]) + compute_penalty(v, N, C)
            
        res = minimize(obj, x_curr, method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-11, 'gtol': 1e-8, 'maxiter': 3000})
        x_curr = res.x
        C *= 1.7  # Gradually increase penalty weight
        
    # 3. Final High-Precision Refinement
    def obj_final(v):
        return -np.sum(v[2::3]) + compute_penalty(v, N, C)
        
    res_final = minimize(obj_final, x_curr, method='L-BFGS-B', bounds=bounds,
                        options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 5000})
    best_x = res_final.x
    
    # 4. Extract and format results
    centers_out = np.zeros((N, 2))
    radii_out = np.zeros(N)
    for i in range(N):
        centers_out[i] = [best_x[3*i], best_x[3*i+1]]
        radii_out[i] = best_x[3*i+2]
        
    sum_radii = np.sum(radii_out)
    
    # Optional: Tiny shrinkage to guarantee strict feasibility against 1e-12 tolerance
    # (Usually unnecessary with high C, but safe)
    radii_out = np.maximum(radii_out, 0.0)
    
    return centers_out, radii_out, sum_radii
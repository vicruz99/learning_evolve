import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    best_r = 0.0
    best_c = None
    
    # Multi-restart optimization to escape local minima
    for seed in range(20):
        np.random.seed(seed)
        
        # 1. Generate hexagonal lattice initial guess
        pts = []
        r_est = 0.1
        s = 2 * r_est
        row = 0
        while len(pts) < n:
            y = row * s * np.sqrt(3)/2
            for col in range(6):
                x = col * s + (row % 2) * (s/2)
                pts.append([x, y])
                if len(pts) >= n: 
                    break
            row += 1
        pts = np.array(pts)
        
        # Normalize and center within the unit square
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = np.max(mx - mn)
        scale = 0.8 / span
        pts = (pts - mn) * scale + (1 - scale * span) / 2
        
        # Ensure strict feasibility for the initial small radius
        pts = np.clip(pts, 0.06, 0.94)
        
        # Concatenate positions and initial radius into optimization variables
        x0 = np.concatenate([pts.flatten(), [0.05]])
        
        def obj(x):
            c = x[:-1].reshape(n, 2)
            r = x[-1]
            pen = 0.0
            
            # Pairwise non-overlap penalty
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt(np.sum((c[i] - c[j])**2))
                    if d < 2*r:
                        pen += (2*r - d)**2
                        
            # Boundary containment penalty
            for i in range(n):
                if c[i,0] < r: pen += (r - c[i,0])**2
                if c[i,0] > 1-r: pen += (c[i,0] - (1-r))**2
                if c[i,1] < r: pen += (r - c[i,1])**2
                if c[i,1] > 1-r: pen += (c[i,1] - (1-r))**2
                
            # Exact penalty formulation: maximize r while penalizing violations
            return -r + 5000.0 * pen
            
        # Optimize positions and radius
        res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-12})
        c_opt = res.x[:-1].reshape(n, 2)
        
        # 2. Compute true feasible radius for converged centers
        min_sep = 2.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((c_opt[i] - c_opt[j])**2))
                if d < min_sep: 
                    min_sep = d
            for val in [c_opt[i,0], 1-c_opt[i,0], c_opt[i,1], 1-c_opt[i,1]]:
                if val < min_sep: 
                    min_sep = val
                    
        r_true = min_sep / 2.0
        
        # Track best configuration found
        if r_true > best_r:
            best_r = r_true
            best_c = c_opt.copy()
            
    radii = np.full(n, best_r)
    return best_c, radii, 26 * best_r
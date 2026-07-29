# sol_000181 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0ae2e142) state=4e2d5937 sum of radii=1.248901 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

N_CIRCLES = 26
PENALTY_COEFF = 8000.0

def compute_objective(vars, n):
    """
    Objective function to minimize: -sum(radii) + penalty for constraint violations.
    """
    x = vars[0:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    val = -np.sum(r)
    pen = 0.0
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    for i in range(n):
        if r[i] > x[i]: 
            pen += (r[i] - x[i])**2
        if r[i] > 1 - x[i]: 
            pen += (r[i] - (1 - x[i]))**2
        if r[i] > y[i]: 
            pen += (r[i] - y[i])**2
        if r[i] > 1 - y[i]: 
            pen += (r[i] - (1 - y[i]))**2
            
    # Overlap constraints: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j]:
                pen += (r[i] + r[j] - dist)**2
                
    return val + PENALTY_COEFF * pen

def run_packing():
    n = N_CIRCLES
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Try multiple random seeds to find the best local optimum
    for seed in [42, 137, 256]:
        rng = np.random.default_rng(seed)
        
        # Initialize on a staggered grid (approximates hexagonal packing)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, 0.05)
        
        idx = 0
        spacing = 0.12 + rng.uniform(0, 0.02)
        for row in range(6):
            y_val = 0.08 + row * spacing * 0.866
            cols = 5 if row % 2 == 0 else 6
            for col in range(cols):
                if idx >= n: 
                    break
                x_val = 0.05 + col * spacing + (row % 2) * spacing * 0.5
                x[idx] = x_val
                y[idx] = y_val
                idx += 1
                
        # Add small random perturbation to break symmetry
        x += rng.uniform(-0.02, 0.02, n)
        y += rng.uniform(-0.02, 0.02, n)
        
        vars0 = np.concatenate([x[:n], y[:n], r[:n]])
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        try:
            res = minimize(compute_objective, vars0, args=(n,), 
                           method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-10})
            
            xc = res.x[0:n]
            yc = res.x[n:2*n]
            rc = res.x[2*n:3*n]
            
            # Strict validity check
            valid = True
            for i in range(n):
                if xc[i] < -1e-12 or xc[i] > 1 + 1e-12 or yc[i] < -1e-12 or yc[i] > 1 + 1e-12 or rc[i] < 0:
                    valid = False
                    break
                for j in range(i + 1, n):
                    d = math.sqrt((xc[i]-xc[j])**2 + (yc[i]-yc[j])**2)
                    if d < rc[i] + rc[j] - 1e-9:
                        valid = False
                        break
                if not valid: 
                    break
                    
            if valid:
                s = np.sum(rc)
                if s > best_sum:
                    best_sum = s
                    best_centers = np.column_stack([xc, yc]).copy()
                    best_radii = rc.copy()
        except Exception:
            continue
            
    # Fallback to initial configuration if optimization failed entirely
    if best_centers is None:
        best_centers = np.column_stack([x[:n], y[:n]])
        best_radii = r[:n]
        
    # Final safety adjustment: guarantee strict validity per validator tolerances
    centers = best_centers
    radii = best_radii
    
    for _ in range(1000):
        ok = True
        for i in range(n):
            cx, cy, cr = centers[i,0], centers[i,1], radii[i]
            if cx < cr - 1e-12 or cx > 1 - cr + 1e-12 or cy < cr - 1e-12 or cy > 1 - cr + 1e-12:
                ok = False
                break
            for j in range(i + 1, n):
                d = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < radii[i] + radii[j] - 1e-12:
                    ok = False
                    break
            if not ok: 
                break
                
        if ok: 
            break
            
        # Slightly shrink radii and project centers to valid region
        radii *= 0.995
        centers[:,0] = np.clip(centers[:,0], radii, 1 - radii)
        centers[:,1] = np.clip(centers[:,1], radii, 1 - radii)
        
    return centers, radii, float(np.sum(radii))

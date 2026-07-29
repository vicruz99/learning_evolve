# sol_000260 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2b83b1fb) state=0bf4e385 sum of radii=2.588664 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Minimize negative sum of radii to maximize total radius."""
    return -np.sum(v[2::3])

def constraints_func(v):
    """
    Returns array of constraint values. All must be >= 0.
    Constraints: boundaries, non-negative radii, non-overlap.
    """
    n = N_CIRCLES
    c = []
    
    # Boundary and positivity constraints
    for i in range(n):
        r = v[3*i+2]
        x = v[3*i]
        y = v[3*i+1]
        c.append(x - r)
        c.append(1.0 - x - r)
        c.append(y - r)
        c.append(1.0 - y - r)
        c.append(r)
        
    # Pairwise non-overlap constraints
    for i in range(n):
        xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
        for j in range(i+1, n):
            xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
            dx = xi - xj
            dy = yi - yj
            c.append(dx*dx + dy*dy - (ri + rj)**2)
            
    return np.array(c)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    
    # --- Initialization: Hexagonal Lattice ---
    centers_init = np.zeros((n, 2))
    radii_init = np.ones(n) * 0.08
    
    row = 0
    idx = 0
    y_spacing = 0.18
    x_spacing = y_spacing * np.sqrt(3) / 2
    
    while idx < n:
        shift = (0.5 * x_spacing) if row % 2 == 1 else 0
        x = 0.1 + shift
        while x <= 0.9 and idx < n:
            centers_init[idx] = [x, 0.1 + row * y_spacing]
            idx += 1
            x += x_spacing
        row += 1
        
    # Flatten to optimization vector [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    bounds = [(0.0, 1.0)] * (3*n)
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    best_v = x0.copy()
    best_obj = objective(x0)
    
    np.random.seed(42)
    # Multiple restarts to escape local minima
    for _ in range(4):
        v0 = x0.copy()
        v0[:2*n] += np.random.uniform(-0.005, 0.005, 2*n)
        v0[2::3] += np.random.uniform(-0.005, 0.005, n)
        v0 = np.clip(v0, 0.0, 1.0)
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-11})
            if res.fun < best_obj:
                best_obj = res.fun
                best_v = res.x
        except Exception:
            pass
            
    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = [best_v[3*i], best_v[3*i+1]]
        radii[i] = best_v[3*i+2]
        
    # Strict validation before return
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1.0 + 1e-9 or y - r < -1e-9 or y + r > 1.0 + 1e-9 or r < 0:
            valid = False
            break
            
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                d2 = (centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2
                if d2 < (radii[i] + radii[j])**2 - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Fallback to safe grid packing if optimizer fails
        idx = 0
        for r in range(5):
            for c in range(6):
                if idx < n:
                    centers[idx] = [(c + 0.5)/6, (r + 0.5)/5]
                    radii[idx] = 0.05
                    idx += 1
        return centers, radii, float(np.sum(radii))
        
    return centers, radii, float(-best_obj)

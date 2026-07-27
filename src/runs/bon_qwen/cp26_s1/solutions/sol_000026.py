# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 27de0ea1) state=c97faf8b sum of radii=2.613222 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(z):
    """Objective function to minimize: negative sum of radii."""
    return -np.sum(z[2::3])

def constraint_boundary(z):
    """Boundary constraints: circles must be inside [0,1]x[0,1]."""
    vals = np.empty(104)
    for i in range(26):
        idx = 3 * i
        x, y, r = z[idx], z[idx+1], z[idx+2]
        vals[4*i] = x - r
        vals[4*i+1] = 1.0 - x - r
        vals[4*i+2] = y - r
        vals[4*i+3] = 1.0 - y - r
    return vals

def constraint_overlap(z):
    """Overlap constraints: squared distance >= squared sum of radii."""
    n = 26
    num_pairs = n * (n - 1) // 2
    vals = np.empty(num_pairs)
    cx = z[0::3]
    cy = z[1::3]
    cr = z[2::3]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dr = cr[i] + cr[j]
            vals[k] = dx*dx + dy*dy - dr*dr
            k += 1
    return vals

def get_init(seed):
    """Generate initial guess using a hexagonal grid pattern perturbed by seed."""
    rng = np.random.RandomState(seed)
    n = 26
    cx, cy = [], []
    # Create a dense hexagonal-like arrangement
    for row in range(6):
        y = 0.08 + row * 0.16
        num = 5 if row % 2 == 0 else 4
        for col in range(num):
            x = 0.08 + col * 0.18 + (0.09 if row % 2 != 0 else 0.0)
            cx.append(x)
            cy.append(y)
    # Fill remaining spots with random positions if needed
    while len(cx) < n:
        cx.append(rng.uniform(0.1, 0.9))
        cy.append(rng.uniform(0.1, 0.9))
    cx = np.array(cx[:n])
    cy = np.array(cy[:n])
    r0 = np.full(n, 0.07)
    z0 = np.empty(3*n)
    z0[0::3] = cx
    z0[1::3] = cy
    z0[2::3] = r0
    return z0

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    best_z = None
    best_obj = np.inf
    
    # Multi-start optimization to avoid local minima
    for seed in range(15):
        z0 = get_init(seed)
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_z = res.x.copy()
        except Exception:
            continue
            
    if best_z is None:
        # Fallback to a valid grid packing
        cx = np.repeat(np.linspace(0.1, 0.9, 5), 5)
        cy = np.tile(np.linspace(0.1, 0.9, 5), 5)
        cx = np.append(cx, 0.5)
        cy = np.append(cy, 0.5)
        r = np.full(26, 0.09)
        return np.column_stack((cx, cy)), r, np.sum(r)
        
    centers = np.column_stack((best_z[0::3], best_z[1::3]))
    radii = np.maximum(best_z[2::3], 0.0)
    
    # Ensure strict feasibility within numerical tolerance
    # Slightly shrink radii if any constraint is marginally violated
    cx, cy = centers[:,0], centers[:,1]
    for _ in range(10):
        max_viol = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                v = radii[i] + radii[j] - d
                if v > max_viol: 
                    max_viol = v
            v1 = radii[i] - cx[i]
            v2 = radii[i] - (1.0 - cx[i])
            v3 = radii[i] - cy[i]
            v4 = radii[i] - (1.0 - cy[i])
            if max(v1, v2, v3, v4) > max_viol:
                max_viol = max(v1, v2, v3, v4)
                
        if max_viol > 1e-9:
            scale = 1.0 - max_viol * 0.5
            radii *= scale
        else:
            break
            
    return centers, radii, np.sum(radii)

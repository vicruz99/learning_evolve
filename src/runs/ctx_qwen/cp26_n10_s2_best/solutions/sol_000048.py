# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 585439f0) state=260fa540 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for efficiency
    xi = x[:, np.newaxis]
    y_i = y[:, np.newaxis]
    ri = r[:, np.newaxis]
    
    dx = xi - x
    dy = y_i - y
    dr = ri + r
    
    dist_sq = dx**2 + dy**2
    r_sum_sq = dr**2
    
    # Upper triangle mask to get unique pairs
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_overlap = (dist_sq - r_sum_sq)[mask]
    
    return np.concatenate([c, c_overlap])

def get_initial(seed):
    """Generate initial configuration using hexagonal lattice with perturbation."""
    np.random.seed(seed)
    r_init = 0.09
    pts = []
    y = r_init
    row = 0
    while len(pts) < N:
        x = r_init if row % 2 == 0 else 2*r_init
        while x < 1 - r_init and len(pts) < N:
            pts.append([x, y])
            x += 2*r_init
        y += r_init * np.sqrt(3)
        row += 1
        
    centers = np.array(pts[:N])
    # Add perturbation to break symmetry and escape lattice minima
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Start with feasible small radii
    r_init_val = np.full(N, 0.04)
    
    return np.concatenate([centers[:,0], centers[:,1], r_init_val])

def run_packing():
    """Optimizes packing of 26 circles in a unit square."""
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_val = -np.inf
    best_v = None
    
    # Multi-start optimization to find global optimum
    for s in range(15):
        x0 = get_initial(s)
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons,
                       options={'maxiter': 12000, 'ftol': 1e-12, 'disp': False})
                       
        # Track best feasible solution
        if -res.fun > best_val:
            if np.all(constraints(res.x) >= -1e-7):
                best_val = -res.fun
                best_v = res.x.copy()
                
    # Extract results
    x = best_v[:N]
    y = best_v[N:2*N]
    r = best_v[2*N:]
    
    # Post-processing to guarantee strict validity per validator rules
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        
    # 2. Enforce non-overlap strictly with safety margin
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j]:
                shrink = (r[i] + r[j] - dist) / 2.0 + 1e-8
                r[i] -= shrink
                r[j] -= shrink
                
    r = np.maximum(r, 0.0)
    centers = np.column_stack([x, y])
    return centers, r, np.sum(r)

# sol_000267 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f6ad2c92) state=4ad952cf sum of radii=2.339304 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(x):
    """Negative sum of radii to maximize total radius."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns array of constraint values that must be >= 0."""
    n = N_CIRCLES
    cx = x[:2*n].reshape(n, 2)
    r = x[2::3]
    
    # Pairwise non-overlap: dist >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
    diff = cx[:, np.newaxis, :] - cx[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e9)  # Ignore self-distances
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p_cons = dist[mask] - r_sum[mask]
    
    # Boundary containment: r <= x <= 1-r  =>  x-r >= 0  and  1-x-r >= 0
    b = np.stack([cx[:,0]-r, 1.0-cx[:,0]-r, cx[:,1]-r, 1.0-cx[:,1]-r], axis=1)
    
    return np.concatenate([p_cons, b.flatten()])

def run_packing():
    np.random.seed(42)
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    start_configs = []
    
    # Config 1: Hexagonal-ish layout 6,5,6,5,4
    rows = [6, 5, 6, 5, 4]
    c1 = []
    for i, count in enumerate(rows):
        y = 0.09 + i * 0.185
        w = (count - 1) * 0.16
        sx = (1.0 - w) / 2.0
        for j in range(count):
            c1.append([sx + j * 0.16, y])
    start_configs.append(c1)
    
    # Config 2: Perturbed 5x5 grid + center
    c2 = []
    for i in range(5):
        for j in range(5):
            c2.append([0.1 + j * 0.2, 0.1 + i * 0.2])
    c2.append([0.5, 0.5])
    start_configs.append(c2)
    
    # Bounds: centers in [0,1], radii in [1e-7, 0.5]
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(1e-7, 0.5) for _ in range(n)]
    cons = {'type': 'ineq', 'fun': constraints}
    
    for cfg in start_configs:
        # Add small random noise to escape symmetry
        cx0 = np.array(cfg) + np.random.randn(n, 2) * 0.002
        r0 = np.full(n, 0.085)
        x0 = np.zeros(3*n)
        x0[:2*n] = cx0.flatten()
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            cx = res.x[:2*n].reshape(n, 2)
            r = res.x[2::3]
            
            # Quick internal validation
            valid = True
            for i in range(n):
                if cx[i,0]-r[i] < -1e-9 or cx[i,0]+r[i] > 1.0+1e-9:
                    valid = False
                if cx[i,1]-r[i] < -1e-9 or cx[i,1]+r[i] > 1.0+1e-9:
                    valid = False
                for j in range(i+1, n):
                    d = np.sqrt(np.sum((cx[i]-cx[j])**2))
                    if d < r[i]+r[j] - 1e-9:
                        valid = False
                        
            if valid and curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = cx.copy()
                best_radii = r.copy()
        except Exception:
            continue
            
    # Fallback to safe grid packing if optimization fails
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_radii = np.full(n, 0.04)
        for i in range(n):
            best_centers[i] = [(i % 5) * 0.2 + 0.1, (i // 5) * 0.2 + 0.1]
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)

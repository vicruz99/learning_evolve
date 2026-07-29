# sol_000027 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000004 (state 5455684e) state=4b055038 sum of radii=2.625974 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars, n):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars, n):
    """
    Compute all inequality constraints g(vars) >= 0.
    Returns a 1D array containing:
    - Boundary constraints (4*n)
    - Squared overlap constraints (n*(n-1)/2)
    """
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    b = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Overlap: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for speed
    dx = x[:, None] - x
    dy = y[:, None] - y
    dr = r[:, None] + r
    
    triu = np.triu_indices(n, k=1)
    dist_sq = dx[triu]**2 + dy[triu]**2
    r_sum_sq = dr[triu]**2
    
    o = dist_sq - r_sum_sq
    return np.concatenate([b, o])

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_vars = None
    
    # Multiple restarts from perturbed hexagonal grids
    for seed in range(40):
        np.random.seed(seed)
        r_init = 0.09 + np.random.uniform(0.0, 0.015)
        centers = []
        y = r_init
        parity = 0
        
        # Generate hexagonal lattice points within the square
        while y <= 1.0 - r_init and len(centers) < n + 5:
            x = r_init if parity == 0 else 2.0 * r_init
            while x <= 1.0 - r_init:
                centers.append([x, y])
                x += 2.0 * r_init
            y += r_init * np.sqrt(3.0)
            parity = 1 - parity
            
        # Fallback to grid if hex generation is sparse
        if len(centers) < n:
            centers = []
            for i in range(6):
                for j in range(5):
                    if len(centers) >= n: break
                    centers.append([0.1 + i * 0.17, 0.1 + j * 0.17])
                    
        centers = np.array(centers[:n])
        # Add noise to break symmetry and help optimizer explore
        centers += np.random.uniform(-0.012, 0.012, centers.shape)
        centers = np.clip(centers, r_init, 1.0 - r_init)
        
        x0 = np.zeros(3 * n)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            vals = constraints(res.x, n)
            # Accept if feasible within tolerance and improves best sum
            if np.all(vals >= -1e-7) and curr_sum > best_sum:
                best_sum = curr_sum
                best_vars = res.x.copy()
        except Exception:
            pass
            
    # Fallback configuration in case all optimizations fail
    if best_vars is None:
        best_vars = np.zeros(3 * n)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    best_vars[3 * idx] = 0.1 + i * 0.2
                    best_vars[3 * idx + 1] = 0.1 + j * 0.2
                    best_vars[3 * idx + 2] = 0.10
                    idx += 1
        if idx < n:
            best_vars[3 * idx] = 0.5
            best_vars[3 * idx + 1] = 0.5
            best_vars[3 * idx + 2] = 0.05

    # Refinement phase: perturb best solution and re-optimize to escape local minima
    for _ in range(5):
        noisy = best_vars + np.random.uniform(-0.001, 0.001, 3 * n)
        for i in range(n):
            r = max(0.0, noisy[3 * i + 2])
            noisy[3 * i] = np.clip(noisy[3 * i], r, 1.0 - r)
            noisy[3 * i + 1] = np.clip(noisy[3 * i + 1], r, 1.0 - r)
        try:
            res = minimize(objective, noisy, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            vals = constraints(res.x, n)
            if np.all(vals >= -1e-7) and curr_sum > best_sum:
                best_sum = curr_sum
                best_vars = res.x.copy()
        except Exception:
            break
            
    # Extract and format output
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_vars[3 * i]
        centers[i, 1] = best_vars[3 * i + 1]
        radii[i] = best_vars[3 * i + 2]
        
    return centers, radii, float(best_sum)

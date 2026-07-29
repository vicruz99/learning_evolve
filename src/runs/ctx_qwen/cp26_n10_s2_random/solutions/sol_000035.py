# sol_000035 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state 58c90071) state=efca61d3 sum of radii=2.623001 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars_flat):
    """Objective function: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def constraints(vars_flat):
    """Computes all boundary and non-overlap constraints."""
    X = vars_flat.reshape(N, 3)
    xs = X[:, 0]
    ys = X[:, 1]
    rs = X[:, 2]
    
    # Pre-allocate constraint array
    con = np.empty(4 * N + N * (N - 1) // 2)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con[:N] = xs - rs
    con[N:2*N] = 1.0 - xs - rs
    con[2*N:3*N] = ys - rs
    con[3*N:4*N] = 1.0 - ys - rs
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    idx = np.triu_indices(N, k=1)
    dx = xs[idx[0]] - xs[idx[1]]
    dy = ys[idx[0]] - ys[idx[1]]
    dr = rs[idx[0]] + rs[idx[1]]
    con[4*N:] = np.hypot(dx, dy) - dr
    
    return con

def get_bounds():
    """Returns variable bounds for x, y, r."""
    b = []
    for _ in range(N):
        b.append((0.0, 1.0))
        b.append((0.0, 1.0))
        b.append((1e-8, 0.5))
    return b

def generate_hex_init(scale=1.0, noise_std=0.0):
    """Generates a feasible initial configuration based on a hexagonal lattice."""
    pts = []
    r_est = 0.095 * scale
    s = 2.0 * r_est
    dy = s * np.sqrt(3.0) / 2.0
    y = r_est
    row = 0
    while len(pts) < N and y + r_est < 1.0:
        x = r_est + (s / 2.0 if row % 2 == 1 else 0.0)
        while x + r_est < 1.0 and len(pts) < N:
            pts.append([x, y])
            x += s
        y += dy
        row += 1
        
    # Fill remaining spots if lattice doesn't yield N points
    while len(pts) < N:
        pts.append([np.random.rand(), np.random.rand()])
        
    arr = np.array(pts[:N])
    if noise_std > 0.0:
        arr += np.random.normal(0, noise_std, arr.shape)
        arr = np.clip(arr, 1e-4, 1 - 1e-4)
        
    init_vec = np.zeros(N * 3)
    for i in range(N):
        init_vec[3 * i] = arr[i, 0]
        init_vec[3 * i + 1] = arr[i, 1]
        init_vec[3 * i + 2] = r_est
    return init_vec

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bnds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_x = None
    best_val = -np.inf
    
    # Phase 1: Multiple restarts from hexagonal grids with varying scales and noise
    for trial in range(20):
        scale = 0.95 + 0.1 * np.random.rand()
        noise = 0.005 + 0.01 * np.random.rand()
        x0 = generate_hex_init(scale, noise)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bnds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun) and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative refinement around the best found solution to escape local minima
    if best_x is not None:
        current_x = best_x.copy()
        for ref_iter in range(12):
            noise_std = 0.002 * (1.0 - ref_iter * 0.06)
            perturbed = current_x.copy()
            for i in range(N):
                perturbed[3*i] += np.random.normal(0, noise_std)
                perturbed[3*i+1] += np.random.normal(0, noise_std)
                perturbed[3*i+2] += np.random.normal(0, noise_std * 0.5)
            # Enforce bounds strictly before optimization
            for i in range(N):
                perturbed[3*i] = np.clip(perturbed[3*i], 0.0, 1.0)
                perturbed[3*i+1] = np.clip(perturbed[3*i+1], 0.0, 1.0)
                perturbed[3*i+2] = np.clip(perturbed[3*i+2], 1e-8, 0.5)
                
            try:
                res = minimize(objective, perturbed, method='SLSQP', bounds=bnds,
                               constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
                if np.isfinite(res.fun) and -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x.copy()
                    current_x = best_x.copy()
            except Exception:
                pass
                
    # Fallback if optimization fails entirely
    if best_x is None:
        best_x = generate_hex_init(1.0, 0.0)
        
    centers = best_x.reshape(N, 3)[:, :2].copy()
    radii = best_x.reshape(N, 3)[:, 2].copy()
    
    # Post-processing: Strictly enforce constraints to guarantee validation passes
    for i in range(N):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        max_r = min(x, 1-x, y, 1-y)
        if r > max_r:
            radii[i] = max_r
            
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            if d < radii[i] + radii[j]:
                # Scale down radii to exactly touch without overlap
                factor = d / (radii[i] + radii[j])
                radii[i] *= factor
                radii[j] *= factor
                
    sum_r = float(np.sum(radii))
    return centers, radii, sum_r

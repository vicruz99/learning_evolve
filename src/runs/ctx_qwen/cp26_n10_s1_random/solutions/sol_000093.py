# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000045 (state 7c76ac7a) state=d9b47e99 sum of radii=1.076815 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_objective(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def compute_constraints(vars):
    """Vectorized inequality constraints >= 0 for valid packing."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    rs = r[:, None] + r[None, :]
    
    iu, ju = np.triu_indices(N, k=1)
    c.append(d[iu, ju] - rs[iu, ju])
    
    return np.concatenate(c)

def generate_hex_init(row_counts):
    """Generates a hexagonal lattice pattern scaled to fit safely inside [0,1]."""
    pts = []
    y = 0.15
    for idx, cnt in enumerate(row_counts):
        shift = 0.1 if idx % 2 == 1 else 0.0
        x = 0.15 + shift
        for _ in range(cnt):
            pts.append([x, y])
            x += 0.18
        y += 0.1 * np.sqrt(3)
    pts = np.array(pts[:N])
    # Ensure strictly inside [0.1, 0.9]
    return np.clip(pts, 0.12, 0.88)

def extract_max_radii(centers):
    """Computes the exact maximum feasible radius for each circle given fixed centers."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        min_d = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_d:
                    min_d = d
        radii[i] = min_d * 0.5
    # Apply tiny safety margin for numerical stability in validation
    return radii * 0.99999

def run_packing():
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    inits = []
    
    # 1. Structured Hexagonal Patterns
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 6, 6, 5, 4],
        [4, 6, 5, 6, 5]
    ]
    for pat in patterns:
        pts = generate_hex_init(pat)
        v0 = np.zeros(3 * N)
        v0[0::3] = pts[:, 0]
        v0[1::3] = pts[:, 1]
        v0[2::3] = 0.04  # Safe initial radius
        inits.append(v0)
        
    # 2. Randomly Perturbed Hex Patterns
    np.random.seed(42)
    for _ in range(12):
        v = inits[0].copy()
        v[0::3] += np.random.uniform(-0.04, 0.04, N)
        v[1::3] += np.random.uniform(-0.04, 0.04, N)
        v[0::3] = np.clip(v[0::3], 0.08, 0.92)
        v[1::3] = np.clip(v[1::3], 0.08, 0.92)
        inits.append(v)
        
    # 3. Regular Grid Fallback
    pts_grid = []
    for i in range(5):
        for j in range(5):
            pts_grid.append([0.1 + j * 0.2, 0.1 + i * 0.2])
    pts_grid.append([0.5, 0.5])
    pts_grid = np.array(pts_grid[:N])
    v_grid = np.zeros(3 * N)
    v_grid[0::3] = pts_grid[:, 0]
    v_grid[1::3] = pts_grid[:, 1]
    v_grid[2::3] = 0.04
    inits.append(v_grid)
    
    # Optimization Loop
    for v0 in inits:
        try:
            res = minimize(compute_objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
        except Exception:
            continue
            
        if res.success or res.fun < -2.5:
            c_opt = res.x[:2*N].reshape(N, 2)
            r_opt = extract_max_radii(c_opt)
            s = np.sum(r_opt)
            
            if s > best_sum:
                best_sum = s
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
                
        # Local perturbation search around optimum to escape shallow local minima
        if res.success:
            v_p = res.x.copy()
            v_p[0::3] += np.random.uniform(-0.005, 0.005, N)
            v_p[1::3] += np.random.uniform(-0.005, 0.005, N)
            v_p[0::3] = np.clip(v_p[0::3], 0.0, 1.0)
            v_p[1::3] = np.clip(v_p[1::3], 0.0, 1.0)
            
            try:
                res2 = minimize(compute_objective, v_p, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
                if res2.success:
                    c_opt2 = res2.x[:2*N].reshape(N, 2)
                    r_opt2 = extract_max_radii(c_opt2)
                    s2 = np.sum(r_opt2)
                    if s2 > best_sum:
                        best_sum = s2
                        best_centers = c_opt2.copy()
                        best_radii = r_opt2.copy()
            except Exception:
                pass

    # Guaranteed valid fallback
    if best_centers is None:
        best_centers = generate_hex_init([5, 6, 5, 6, 4])
        best_radii = extract_max_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)

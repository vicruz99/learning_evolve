# sol_000025 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000015 (state cc21d5f7) state=d15e4e7a sum of radii=2.626678 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def constraint_func(v):
    """
    Computes inequality constraints: boundary containment and pairwise separation.
    All constraints are formulated as g(v) >= 0.
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    # Upper triangular mask to avoid duplicates and self-comparison
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c.append(dist2[mask] - rs[mask]**2)
    
    return np.concatenate(c)

def get_init_hex():
    """Generates a hexagonal lattice initialization."""
    pts = []
    r_est = 0.09
    y = r_est
    row = 0
    while len(pts) < N:
        x_off = (row % 2) * r_est
        x = r_est + x_off
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    return np.array(pts[:N])

def get_init_grid():
    """Generates a perturbed square grid initialization with one extra circle."""
    pts = []
    # Standard 5x5 grid
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    # 26th circle placed in a central gap
    pts.append([0.2, 0.35])
    return np.array(pts[:N])

def run_packing():
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    bases = [get_init_hex(), get_init_grid()]
    
    # Multiple restarts to escape local minima
    for seed in range(50):
        np.random.seed(seed)
        base = bases[seed % 2]
        
        # Perturb base layout
        pts = base + np.random.uniform(-0.05, 0.05, (N, 2))
        pts = np.clip(pts, 0.02, 0.98)
        
        # Compute strictly feasible initial radii
        r_safe = np.full(N, 0.001)
        for i in range(N):
            max_r = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d * 0.5 < max_r:
                    max_r = d * 0.5
            r_safe[i] = max_r * 0.88
            
        # Flatten to optimization vector: [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = pts[i, 0]
            x0[3*i+1] = pts[i, 1]
            x0[3*i+2] = r_safe[i]
            
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 2500, 'ftol': 1e-13, 'disp': False}
            )
            
            # Verify constraint satisfaction and track best
            if res.success:
                c_val = constraint_func(res.x)
                if np.min(c_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                        best_radii = res.x[2::3]
        except Exception:
            continue

    # High-precision refinement on the best configuration found
    if best_centers is not None:
        x0_best = np.concatenate([best_centers.flatten(), best_radii])
        try:
            res_final = minimize(
                objective, 
                x0_best, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
            )
            if res_final.success:
                c_val = constraint_func(res_final.x)
                if np.min(c_val) >= -1e-8:
                    best_sum = -res_final.fun
                    best_centers = np.column_stack((res_final.x[0::3], res_final.x[1::3]))
                    best_radii = res_final.x[2::3]
        except Exception:
            pass
            
    # Fallback (should not be reached given robust initialization)
    if best_centers is None:
        best_centers = np.random.uniform(0.1, 0.9, (N, 2))
        best_radii = np.full(N, 0.01)
        best_sum = np.sum(best_radii)
        
    # Ensure non-negative radii against numerical drift
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, float(best_sum)

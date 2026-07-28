# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000052 (state e51e4326) state=863498cb sum of radii=1.674000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_array, n, triu_idx):
    """Minimize negative sum of radii"""
    return -np.sum(vars_array[:n])

def constraint_fun(vars_array, n, triu_idx):
    """
    Computes pairwise non-overlap constraints:
    dist_sq(i,j) - (r_i + r_j)^2 >= 0
    Boundary constraints are satisfied analytically by the parameterization.
    """
    r = vars_array[:n]
    u = vars_array[n:2*n]
    v = vars_array[2*n:3*n]
    
    x = r + (1.0 - 2.0*r)*u
    y = r + (1.0 - 2.0*r)*v
    
    diff_x = x[:, np.newaxis] - x[np.newaxis, :]
    diff_y = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = diff_x**2 + diff_y**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    return dist_sq[triu_idx] - r_sum_sq[triu_idx]

def generate_hex_init(n, r, angle_deg=0, seed=None):
    """Generates a rotated and centered hexagonal grid initialization."""
    pts = []
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    y = -0.2
    while len(pts) < n + 10:
        x = -0.2
        shift = (int((y - (-0.2)) / dy) % 2) * dx / 2.0
        while x < 1.2:
            pts.append([x + shift, y])
            x += dx
        y += dy
        
    pts = np.array(pts)
    if angle_deg != 0:
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts = (R @ pts.T).T
        
    # Center and scale to fit comfortably inside [0,1]
    pts -= pts.mean(axis=0)
    max_ext = np.max(np.abs(pts), axis=0)
    scale = 0.4 / max(max_ext[0], max_ext[1])
    pts *= scale
    pts += 0.5
    
    # Convert to parameterization coordinates
    denom = 1.0 - 2.0*r
    u = (pts[:, 0] - r) / denom
    v = (pts[:, 1] - r) / denom
    
    # Add controlled noise to break symmetries
    if seed is not None:
        np.random.seed(seed)
        u += np.random.uniform(-0.02, 0.02, n)
        v += np.random.uniform(-0.02, 0.02, n)
        
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    return np.concatenate([np.full(n, r), u[:n], v[:n]])

def run_packing():
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    bounds = [(1e-4, 0.5)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': constraint_fun, 'args': (n, triu_idx)}
    
    best_vars = None
    best_sum = -np.inf
    
    configs = []
    # Diverse initializations with rotations to escape square-symmetry traps
    for ang in [0, 15, 30]:
        for r_init in [0.09, 0.10]:
            for seed in [0, 1, 2]:
                try:
                    configs.append(generate_hex_init(n, r_init, ang, seed))
                except Exception:
                    pass
                    
    # Add fully random configurations for robustness
    np.random.seed(42)
    for _ in range(5):
        configs.append(np.concatenate([
            np.random.uniform(0.09, 0.11, n),
            np.random.uniform(0.0, 1.0, n),
            np.random.uniform(0.0, 1.0, n)
        ]))
        
    # Multi-start optimization
    for x0 in configs:
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                args=(n, triu_idx),
                options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                c_vals = constraint_fun(res.x, n, triu_idx)
                if np.min(c_vals) >= -1e-5:
                    s = np.sum(res.x[:n])
                    if s > best_sum:
                        best_sum = s
                        best_vars = res.x.copy()
        except Exception:
            pass
            
    if best_vars is None:
        best_vars = configs[0]
        
    # Decode optimized positions
    r = best_vars[:n]
    u = best_vars[n:2*n]
    v = best_vars[2*n:3*n]
    x = r + (1.0 - 2.0*r)*u
    y = r + (1.0 - 2.0*r)*v
    centers = np.column_stack((x, y))
    
    # Post-processing: Compute exact maximal feasible radii for the optimized centers.
    # This breaks artificial equality constraints and typically increases the sum.
    new_radii = np.zeros(n)
    for i in range(n):
        min_d = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        for j in range(n):
            if i != j:
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < min_d:
                    min_d = d
        new_radii[i] = min_d / 2.0
        
    # Apply tiny safety margin to guarantee strict validator compliance
    new_radii *= 0.999995
    
    # Strict validation check
    valid = True
    for i in range(n):
        if x[i] - new_radii[i] < -1e-12 or x[i] + new_radii[i] > 1 + 1e-12 or \
           y[i] - new_radii[i] < -1e-12 or y[i] + new_radii[i] > 1 + 1e-12:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                dist = np.hypot(x[i]-x[j], y[i]-y[j])
                if dist < new_radii[i] + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid: break
            
    if valid:
        return centers, new_radii, float(np.sum(new_radii))
        
    # Fallback: scale down original radii to guarantee validity
    scale = 1.0
    for i in range(n):
        scale = min(scale, x[i]/r[i], (1-x[i])/r[i], y[i]/r[i], (1-y[i])/r[i])
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            if r[i]+r[j] > 1e-9:
                scale = min(scale, d/(r[i]+r[j]))
                
    final_r = r * scale * 0.99999
    return centers, final_r, float(np.sum(final_r))

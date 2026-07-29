# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4ac25994) state=32ca1949 sum of radii=2.547202 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Top-level factory functions to avoid lambdas and closures
def make_x_min_constraint(i, N):
    def func(v):
        return v[2 * i] - v[2 * N + i]
    return func

def make_x_max_constraint(i, N):
    def func(v):
        return 1.0 - v[2 * i] - v[2 * N + i]
    return func

def make_y_min_constraint(i, N):
    def func(v):
        return v[2 * i + 1] - v[2 * N + i]
    return func

def make_y_max_constraint(i, N):
    def func(v):
        return 1.0 - v[2 * i + 1] - v[2 * N + i]
    return func

def make_r_min_constraint(i, N):
    def func(v):
        return v[2 * N + i] - 1e-7
    return func

def make_pair_constraint(i, j, N):
    def func(v):
        xi, yi, ri = 2 * i, 2 * i + 1, 2 * N + i
        xj, yj, rj = 2 * j, 2 * j + 1, 2 * N + j
        return (v[xi] - v[xj])**2 + (v[yi] - v[yj])**2 - (v[ri] + v[rj])**2
    return func

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    
    # 1. Initial Layout: Hexagonal Packing
    # Row counts: 6, 5, 6, 5, 4 sums to 26
    row_counts = [6, 5, 6, 5, 4]
    r_init = 0.08
    h = np.sqrt(3) * r_init
    
    centers = []
    y = r_init
    for row_idx, cnt in enumerate(row_counts):
        for k in range(cnt):
            if row_idx % 2 == 0:
                x = r_init + k * 2 * r_init
            else:
                x = 2 * r_init + k * 2 * r_init
            centers.append([x, y])
        y += h
        
    centers = np.array(centers)
    
    # Normalize to [0, 1] with padding to allow room for expansion
    c_min = centers.min(axis=0)
    c_max = centers.max(axis=0)
    c_range = c_max - c_min
    scale = 0.85 / c_range.max()
    centers = (centers - c_min) * scale + (1.0 - scale) / 2.0
    radii = np.full(N, r_init * 0.4)
    
    # 2. Prepare Optimization
    # Variable vector: [x0, y0, x1, y1, ..., x25, y25, r0, r1, ..., r25]
    x0 = np.concatenate([centers.flatten(), radii])
    
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': make_x_min_constraint(i, N)})
        cons.append({'type': 'ineq', 'fun': make_x_max_constraint(i, N)})
        cons.append({'type': 'ineq', 'fun': make_y_min_constraint(i, N)})
        cons.append({'type': 'ineq', 'fun': make_y_max_constraint(i, N)})
        cons.append({'type': 'ineq', 'fun': make_r_min_constraint(i, N)})
        
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq', 'fun': make_pair_constraint(i, j, N)})
            
    # Bounds for variables
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # Objective: Maximize sum of radii => Minimize -sum(radii)
    def objective(v):
        return -np.sum(v[2 * N:])
        
    # Run SLSQP optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                   
    # Extract results
    final_centers = res.x[:2 * N].reshape((N, 2))
    final_radii = res.x[2 * N:]
    
    # Safety clamp and validation check
    final_radii = np.maximum(final_radii, 0.0)
    final_centers = np.clip(final_centers, 0.0, 1.0)
    
    # Re-validate locally to ensure no numerical drift violates constraints strictly
    # If invalid, fallback to a scaled down version of the result
    valid = True
    for i in range(N):
        if final_centers[i, 0] - final_radii[i] < -1e-9 or final_centers[i, 0] + final_radii[i] > 1 + 1e-9:
            valid = False
        if final_centers[i, 1] - final_radii[i] < -1e-9 or final_centers[i, 1] + final_radii[i] > 1 + 1e-9:
            valid = False
            
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                if dist < final_radii[i] + final_radii[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Fallback: shrink radii slightly until valid
        scale_factor = 0.99
        for _ in range(50):
            final_radii *= scale_factor
            is_valid = True
            for i in range(N):
                if final_centers[i, 0] - final_radii[i] < 0 or final_centers[i, 0] + final_radii[i] > 1:
                    is_valid = False
                if final_centers[i, 1] - final_radii[i] < 0 or final_centers[i, 1] + final_radii[i] > 1:
                    is_valid = False
            if is_valid:
                break
                
    return final_centers, final_radii, float(np.sum(final_radii))

# sol_000130 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0b92a944) state=b64e207e sum of radii=2.591391 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any(): return False
    if np.isnan(radii).any(): return False

    for i in range(n):
        if radii[i] < 0: return False
        elif np.isnan(radii[i]): return False
        
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def objective(x, n):
    # x contains [x1, y1, r1, x2, y2, r2, ...]
    # We want to maximize sum of radii, so minimize -sum(r)
    radii = x[2::3]
    return -np.sum(radii)

def boundary_constraints(x, n):
    # x_i >= r_i, x_i + r_i <= 1, y_i >= r_i, y_i + r_i <= 1
    cons = []
    for i in range(n):
        idx = i * 3
        xi = x[idx]
        yi = x[idx+1]
        ri = x[idx+2]
        
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3] - v[i*3+2]})
        # 1 - (x + r) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - (v[i*3] + v[i*3+2])})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3+1] - v[i*3+2]})
        # 1 - (y + r) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - (v[i*3+1] + v[i*3+2])})
    return cons

def overlap_constraints(x, n):
    # dist(i, j) >= ri + rj
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = i * 3
            idx_j = j * 3
            
            def make_constraint(i_idx, j_idx):
                def constraint(v):
                    xi, yi = v[i_idx], v[i_idx+1]
                    xj, yj = v[j_idx], v[j_idx+1]
                    ri, rj = v[i_idx+2], v[j_idx+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    # dist >= ri + rj  <=>  dist^2 >= (ri + rj)^2 (since dist, radii >= 0)
                    # Using squared distance avoids sqrt, but (ri+rj)^2 is non-linear.
                    # Let's stick to dist - (ri+rj) >= 0 for stability or use squared if careful.
                    # sqrt is fine for scipy.
                    dist = math.sqrt(dist_sq)
                    return dist - (ri + rj)
                return constraint

            cons.append({'type': 'ineq', 'fun': make_constraint(idx_i, idx_j)})
    return cons

def run_packing():
    np.random.seed(42)
    n = 26
    best_sum = -1.0
    best_solution = None
    best_valid = False

    # Initial guesses
    # 1. Grid initialization
    # 2. Random initialization
    
    initial_configs = []
    
    # Config 1: 5x5 Grid + 1 random
    # 25 circles in grid
    grid_r = 0.1
    centers_grid = []
    for r_idx in range(5):
        for c_idx in range(5):
            cx = 0.1 + c_idx * 0.2
            cy = 0.1 + r_idx * 0.2
            centers_grid.append([cx, cy, grid_r])
    # 26th circle at center? Center is occupied. Put at (0.5, 0.1) -> occupied.
    # Put at (0.05, 0.05) with small radius?
    centers_grid.append([0.5, 0.5, 0.01]) # Wait, (0.5, 0.5) is in grid.
    # The grid covers 0.1, 0.3, 0.5, 0.7, 0.9.
    # (0.5, 0.5) is in grid.
    # Let's add one at (0.05, 0.5) or similar gap.
    centers_grid.pop() # Remove last added if it was bad, but I added (0.5, 0.5).
    # Let's re-generate properly.
    centers_grid = []
    pts = [0.1, 0.3, 0.5, 0.7, 0.9]
    for y in pts:
        for x in pts:
            centers_grid.append([x, y, grid_r])
    # Add 26th at (0.5, 0.05) - distance to (0.5, 0.1) is 0.05. r=0.025.
    centers_grid.append([0.5, 0.05, 0.025])
    
    initial_configs.append(np.array(centers_grid).flatten())
    
    # Config 2: Random
    for _ in range(3):
        x_rand = np.random.rand(n) * 0.8 + 0.1
        y_rand = np.random.rand(n) * 0.8 + 0.1
        r_rand = np.random.rand(n) * 0.05 + 0.05
        init_vec = np.zeros(n * 3)
        for i in range(n):
            init_vec[3*i] = x_rand[i]
            init_vec[3*i+1] = y_rand[i]
            init_vec[3*i+2] = r_rand[i]
        initial_configs.append(init_vec)

    for init_x in initial_configs:
        # Bounds: x, y in [0, 1], r >= 0
        bounds = [(0, 1), (0, 1), (0, 0.5)] * n
        
        cons = boundary_constraints(init_x, n)
        cons.extend(overlap_constraints(init_x, n))
        
        try:
            res = minimize(
                objective, 
                init_x, 
                args=(n,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < -2.0: # Check if it found something reasonable
                sum_r = -res.fun
                centers = res.x.reshape(-1, 3)[:, :2]
                radii = res.x.reshape(-1, 3)[:, 2]
                
                # Validate
                if validate_packing(centers, radii):
                    if sum_r > best_sum:
                        best_sum = sum_r
                        best_solution = (centers, radii, sum_r)
                        best_valid = True
        except Exception as e:
            print(f"Optimization failed: {e}")

    # If no valid solution found, fallback to a known valid grid
    if not best_valid:
        pts = [0.1, 0.3, 0.5, 0.7, 0.9]
        centers = []
        radii = []
        for y in pts:
            for x in pts:
                centers.append([x, y])
                radii.append(0.1)
        centers.append([0.5, 0.05])
        radii.append(0.025) # Small circle in gap
        centers = np.array(centers)
        radii = np.array(radii)
        best_sum = np.sum(radii)
        best_solution = (centers, radii, best_sum)

    return best_solution

# Example usage
# run_packing()

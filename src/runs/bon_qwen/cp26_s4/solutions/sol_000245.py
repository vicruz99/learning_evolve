# sol_000245 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e7c70ed6) state=473d2b1a sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars, n):
    """Objective function: minimize negative sum of radii."""
    radii = vars[2::3]
    return -np.sum(radii)

def constraint_boundary(vars, n):
    """Boundary constraints: circles inside [0,1]^2."""
    cons = []
    for i in range(n):
        x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
        cons.extend([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
    return np.array(cons)

def constraint_overlap(vars, n):
    """Overlap constraints: dist_sq >= (r_i + r_j)^2."""
    cons = []
    for i in range(n):
        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
        for j in range(i + 1, n):
            xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            cons.append(dist_sq - (ri + rj) ** 2)
    return np.array(cons)

def generate_hex_guess(n, seed):
    """Generate an initial hexagonal grid configuration."""
    np.random.seed(seed)
    # Row configuration summing to 26
    row_counts = [5, 6, 5, 6, 4]
    r_init = 0.06
    dy = r_init * np.sqrt(3)
    dx = 2.0 * r_init
    
    centers = []
    y_pos = r_init
    for r_idx, count in enumerate(row_counts):
        x_start = r_init + (dx * 0.5 if r_idx % 2 == 1 else 0)
        for _ in range(count):
            centers.append([x_start, y_pos])
            x_start += dx
        y_pos += dy
        
    # Fallback if pattern doesn't sum to n (should be exact)
    while len(centers) < n:
        centers.append([np.random.rand(), np.random.rand()])
    centers = np.array(centers[:n])
    
    # Flatten to [x1, y1, r1, x2, y2, r2, ...]
    guess = np.zeros(3 * n)
    guess[0::3] = centers[:, 0]
    guess[1::3] = centers[:, 1]
    guess[2::3] = r_init
    return guess

def run_packing():
    n = 26
    best_score = -1e9
    best_centers = None
    best_radii = None
    
    # Variable bounds: x,y in [0,1], r in [small_positive, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Constraint definitions passing n explicitly
    cons_list = [
        {'type': 'ineq', 'fun': constraint_boundary, 'args': (n,)},
        {'type': 'ineq', 'fun': constraint_overlap, 'args': (n,)}
    ]
    
    for seed in range(5):
        x0 = generate_hex_guess(n, seed)
        # Add small perturbation to help escape symmetry/local minima
        x0 += np.random.normal(0, 0.005, size=x0.shape)
        # Clip to valid bounds
        x0[:2*n] = np.clip(x0[:2*n], 0.05, 0.95)
        x0[2*n:] = np.clip(x0[2*n:], 1e-4, 0.4)
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds, 
                           constraints=cons_list, options={'maxiter': 150, 'ftol': 1e-9, 'disp': False})
            
            if res.fun < best_score:
                cx = res.x[:2*n].reshape(n, 2)
                rx = res.x[2*n:]
                
                # Strict validation check
                valid = True
                for i in range(n):
                    if (cx[i, 0] - rx[i] < -1e-8 or cx[i, 0] + rx[i] > 1 + 1e-8 or
                        cx[i, 1] - rx[i] < -1e-8 or cx[i, 1] + rx[i] > 1 + 1e-8):
                        valid = False
                        break
                
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            dist = np.hypot(cx[i, 0] - cx[j, 0], cx[i, 1] - cx[j, 1])
                            if dist < rx[i] + rx[j] - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid:
                    best_score = res.fun
                    best_centers = cx.copy()
                    best_radii = rx.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = np.random.rand(n, 2) * 0.8 + 0.1
        best_radii = np.full(n, 0.01)
        
    # Apply a tiny scaling factor to guarantee validation passes despite floating point accumulation
    safety_factor = 0.9999999
    best_radii *= safety_factor
    best_centers[:, 0] = best_centers[:, 0] * (1 - best_radii) + best_radii * 0.5 # keep away from exact edges if needed, but direct is fine
    # Actually, just scaling radii is safest and simplest
    best_radii *= safety_factor 

    return best_centers, best_radii, np.sum(best_radii)

# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05693c56) state=8b34c44a sum of radii=2.629270 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def constraint_func(v):
    """
    Computes constraint values that must be >= 0.
    v: flattened array [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    n = v.shape[0] // 3
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    violations = []
    
    # Boundary constraints: 0 <= x-r, 1-x-r, 0 <= y-r, 1-y-r
    for i in range(n):
        violations.append(c[i, 0] - r[i])
        violations.append(1.0 - c[i, 0] - r[i])
        violations.append(c[i, 1] - r[i])
        violations.append(1.0 - c[i, 1] - r[i])
        
    # Pairwise non-overlap constraints: dist - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            violations.append(dist - r[i] - r[j])
            
    return np.array(violations)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)
    
    # 1. Initialize centers in a shifted hexagonal grid
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05)
    idx = 0
    for row in range(6):
        cols = 5 if row < 5 else 1
        for col in range(cols):
            if idx >= n: 
                break
            x = 0.15 + col * 0.16
            y = 0.15 + row * 0.16
            if row % 2 == 1:
                x += 0.08
            centers[idx] = [x, y]
            idx += 1
            
    # Ensure initial configuration is strictly inside the square
    centers = np.clip(centers, 0.05, 0.95)
    v0 = np.hstack([centers.ravel(), radii])
    
    def objective(v):
        # Maximize sum of radii -> minimize negative sum
        return -np.sum(v[2*n:])
        
    # Nonlinear constraints: all values from constraint_func must be >= 0
    cons = NonlinearConstraint(constraint_func, 0.0, np.inf)
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    best_v = v0.copy()
    best_val = objective(v0)
    
    # 2. Multi-start optimization to avoid local minima
    for restart in range(5):
        v_init = v0.copy()
        v_init[:2*n] += np.random.uniform(-0.02, 0.02, 2*n)
        v_init[:2*n] = np.clip(v_init[:2*n], 0.02, 0.98)
        v_init[2*n:] = np.full(n, 0.05)
        
        try:
            res = minimize(objective, v_init, method='trust-constr', 
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 1000, 'verbose': 0})
            if not np.isnan(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_v = res.x
        except Exception:
            continue
            
    # 3. Extract and validate final configuration
    c_final = best_v[:2*n].reshape(n, 2)
    r_final = best_v[2*n:]
    r_final = np.maximum(r_final, 0.0)
    
    sum_radii = float(np.sum(r_final))
    return c_final, r_final, sum_radii

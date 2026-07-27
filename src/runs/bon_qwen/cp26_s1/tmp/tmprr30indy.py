import numpy as np
from scipy.optimize import minimize

def compute_constraints(v):
    """
    Computes all constraint values for the packing problem.
    v is a flattened array of shape (3*n,) containing [x0, y0, r0, x1, y1, r1, ...]
    Returns an array of constraint values. All must be >= 0 for feasibility.
    """
    n = 26
    xs = v[0::3]
    ys = v[1::3]
    rs = v[2::3]
    
    # Total number of constraints: 4*n (boundaries) + n*(n-1)/2 (non-overlap)
    n_cons = 4 * n + n * (n - 1) // 2
    cons = np.empty(n_cons)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    cons[0:n] = xs - rs
    cons[n:2*n] = 1.0 - xs - rs
    cons[2*n:3*n] = ys - rs
    cons[3*n:4*n] = 1.0 - ys - rs
    
    # Non-overlap constraints: distance(i,j) >= r_i + r_j
    idx = 4 * n
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.hypot(dx, dy)
            cons[idx] = dist - (rs[i] + rs[j])
            idx += 1
            
    return cons

def objective_func(v):
    """Objective: minimize negative sum of radii (i.e., maximize sum of radii)"""
    n = 26
    return -np.sum(v[2::3])

def run_packing():
    n = 26
    best_sum_r = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for x, y in [0, 1] and r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))
        
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Multi-start optimization to escape local minima
    for trial in range(7):
        rng = np.random.default_rng(trial * 42)
        # Initialize centers randomly, avoiding edges to ensure initial feasibility with small radii
        centers = rng.uniform(0.15, 0.85, size=(n, 2))
        init_radii = np.full(n, 0.03)
        
        x0 = np.zeros(3 * n)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = init_radii
        
        try:
            res = minimize(
                objective_func, 
                x0, 
                bounds=bounds, 
                constraints=cons, 
                method='SLSQP', 
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            cur_sum_r = -res.fun
            if cur_sum_r > best_sum_r:
                best_sum_r = cur_sum_r
                best_centers = res.x[:2 * n].reshape((n, 2))
                best_radii = res.x[2 * n:]
        except Exception:
            continue
            
    # Final safety adjustments to ensure strict feasibility
    if best_centers is not None:
        best_centers[:, 0] = np.clip(best_centers[:, 0], best_radii, 1.0 - best_radii)
        best_centers[:, 1] = np.clip(best_centers[:, 1], best_radii, 1.0 - best_radii)
        best_radii = np.maximum(best_radii, 0.0)
    else:
        # Fallback configuration if optimization completely fails
        best_centers = np.tile([0.5, 0.5], (n, 1))
        best_radii = np.full(n, 0.0)
        best_sum_r = 0.0
        
    return best_centers, best_radii, best_sum_r
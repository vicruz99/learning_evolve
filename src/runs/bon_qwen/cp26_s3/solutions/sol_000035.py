# sol_000035 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f550adc) state=bb343d81 sum of radii=2.581049 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Minimize negative sum of radii (equivalent to maximizing sum of radii)"""
    return -np.sum(vars[2 * N:])

def constraints(vars):
    """
    Returns concatenated array of all inequality constraints.
    Constraints are:
    1. x_i - r_i >= 0
    2. 1 - x_i - r_i >= 0
    3. y_i - r_i >= 0
    4. 1 - y_i - r_i >= 0
    5. (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0 for i < j
    """
    c = vars[:2 * N].reshape(N, 2)
    r = vars[2 * N:]
    x, y = c[:, 0], c[:, 1]
    
    con = []
    # Boundary constraints
    con.append(x - r)
    con.append(1 - x - r)
    con.append(y - r)
    con.append(1 - y - r)
    
    # Pairwise non-overlap constraints (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    # Distance squared minus sum of radii squared, upper triangle only
    con.append((dx**2 + dy**2 - dr**2)[np.triu_indices(N, k=1)])
    
    return np.concatenate(con)

def run_packing():
    best_val = -np.inf
    best_res = None
    
    # Run multiple restarts to avoid local optima
    for i in range(20):
        np.random.seed(i * 17)
        
        # Generate initial configuration
        if i % 2 == 0:
            # Hexagonal lattice initialization
            r0 = 0.08
            xs, ys = [], []
            row = 0
            while len(xs) < N:
                col = 0
                while len(xs) < N:
                    # Hexagonal offset
                    x = col * 2 * r0 + (r0 if row % 2 else 0)
                    y = row * np.sqrt(3) * r0
                    xs.append(x)
                    ys.append(y)
                    col += 1
                row += 1
            xs = np.array(xs)
            ys = np.array(ys)
            # Normalize to fit comfortably inside [0,1]
            xs = (xs - xs.min()) / (xs.max() - xs.min()) * 0.8 + 0.1
            ys = (ys - ys.min()) / (ys.max() - ys.min()) * 0.8 + 0.1
            c = np.column_stack((xs, ys))
        else:
            # Random initialization
            c = np.random.uniform(0.15, 0.85, size=(N, 2))
            
        # Small random perturbation to radii to break symmetry
        r_init = np.full(N, 0.06) + np.random.uniform(-0.01, 0.01, N)
        r_init = np.maximum(r_init, 0.01)
        
        x0 = np.hstack([c.flatten(), r_init])
        
        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraints},
                       options={'maxiter': 4000, 'ftol': 1e-12})
        
        # Track best solution
        if res.success:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_res = res
                
    if best_res is not None:
        centers = best_res.x[:2 * N].reshape(N, 2)
        radii = best_res.x[2 * N:]
        
        # Ensure numerical validity
        radii = np.maximum(radii, 0.0)
        centers = np.clip(centers, 0.0, 1.0)
        
        return centers, radii, np.sum(radii)
    else:
        # Fallback (should not be reached)
        return np.zeros((N, 2)), np.zeros(N), 0.0

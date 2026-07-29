# sol_000355 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b8d6b6a1) state=026beec7 sum of radii=2.598024 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
PAIR_IDX = np.triu_indices(N_CIRCLES, k=1)

def objective(v):
    """Minimize negative sum of radii"""
    return -np.sum(v[2::3])

def constraints_func(v):
    """
    Returns array of constraint values. All must be >= 0 for feasibility.
    Structure: [boundary_constraints, pairwise_distance_constraints]
    """
    X = v[0::3]
    Y = v[1::3]
    R = v[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = X - R
    c2 = 1.0 - X - R
    c3 = Y - R
    c4 = 1.0 - Y - R
    
    # Pairwise distance constraints: dist(i,j) >= r_i + r_j
    DX = X[:, None] - X[None, :]
    DY = Y[:, None] - Y[None, :]
    Dist = np.sqrt(DX**2 + DY**2)
    SumR = R[:, None] + R[None, :]
    
    c_pair = Dist[PAIR_IDX] - SumR[PAIR_IDX]
    
    return np.concatenate([c1, c2, c3, c4, c_pair])

def get_initial_guess(seed, use_hex=True):
    """Generate a feasible initial configuration"""
    rng = np.random.default_rng(seed)
    n = N_CIRCLES
    r_init = 0.07
    
    if use_hex:
        counts = [6, 5, 6, 5, 4]
        X0 = []
        Y0 = []
        dy = np.sqrt(3) * r_init
        for row, cnt in enumerate(counts):
            y = r_init + row * dy
            total_w = (cnt - 1) * 2 * r_init
            x_start = (1.0 - total_w) / 2.0
            for col in range(cnt):
                x = x_start + col * 2 * r_init
                X0.append(x)
                Y0.append(y)
        X0 = np.array(X0) + rng.uniform(-0.02, 0.02, n)
        Y0 = np.array(Y0) + rng.uniform(-0.02, 0.02, n)
    else:
        X0 = rng.uniform(r_init + 0.01, 1.0 - r_init - 0.01, n)
        Y0 = rng.uniform(r_init + 0.01, 1.0 - r_init - 0.01, n)
        
    R0 = np.full(n, r_init)
    X0 = np.clip(X0, r_init + 1e-4, 1.0 - r_init - 1e-4)
    Y0 = np.clip(Y0, r_init + 1e-4, 1.0 - r_init - 1e-4)
    return np.concatenate([X0, Y0, R0])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    best_v = None
    best_sum = -np.inf
    
    # Multi-start optimization to escape local minima
    for i in range(30):
        use_hex = (i % 2 == 0)
        v0 = get_initial_guess(i, use_hex)
        
        res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
        
        if res.success:
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_v = res.x
                
    # Fallback if optimization fails (should not happen with feasible starts)
    if best_v is None:
        best_v = get_initial_guess(0, True)
        
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = best_v[2::3]
    radii = np.maximum(radii, 1e-9)  # Ensure non-negative radii
    
    return centers, radii, np.sum(radii)

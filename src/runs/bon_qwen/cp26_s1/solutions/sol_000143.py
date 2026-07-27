# sol_000143 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a66096c7) state=1469003d sum of radii=2.594691 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26

def objective(vars):
    """
    Objective function: Minimize negative sum of radii.
    vars layout: [x1..xN, y1..yN, r1..rN]
    """
    r = vars[2*N:3*N]
    return -np.sum(r)

def con_x_lb(vars):
    """Constraint: x_i >= r_i"""
    return vars[0:N] - vars[2*N:3*N]

def con_x_ub(vars):
    """Constraint: x_i <= 1 - r_i  =>  x_i + r_i <= 1"""
    return 1.0 - (vars[0:N] + vars[2*N:3*N])

def con_y_lb(vars):
    """Constraint: y_i >= r_i"""
    return vars[N:2*N] - vars[2*N:3*N]

def con_y_ub(vars):
    """Constraint: y_i <= 1 - r_i  =>  y_i + r_i <= 1"""
    return 1.0 - (vars[N:2*N] + vars[2*N:3*N])

def con_r_lb(vars):
    """Constraint: r_i >= 0"""
    return vars[2*N:3*N]

class OverlapConstraint:
    """
    Constraint: dist(i, j)^2 >= (r_i + r_j)^2
    """
    def __init__(self, i, j):
        self.i = i
        self.j = j
    
    def __call__(self, vars):
        x = vars[0:N]
        y = vars[N:2*N]
        r = vars[2*N:3*N]
        
        dx = x[self.i] - x[self.j]
        dy = y[self.i] - y[self.j]
        
        r_sum = r[self.i] + r[self.j]
        return dx*dx + dy*dy - r_sum*r_sum

def run_packing():
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    # Using bounds helps restrict the search space
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    constraints = []
    constraints.append({'type': 'ineq', 'fun': con_x_lb})
    constraints.append({'type': 'ineq', 'fun': con_x_ub})
    constraints.append({'type': 'ineq', 'fun': con_y_lb})
    constraints.append({'type': 'ineq', 'fun': con_y_ub})
    constraints.append({'type': 'ineq', 'fun': con_r_lb})
    
    # Add pairwise non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq', 'fun': OverlapConstraint(i, j)})

    best_res = None
    best_obj = np.inf  # We are minimizing negative sum, so lower is better
    
    def try_optimize(x0, seed=None):
        nonlocal best_res, best_obj
        if seed is not None:
            np.random.seed(seed)
        
        # Ensure initial radii are non-negative
        x0_safe = x0.copy()
        if np.any(x0_safe[2*N:] < 0):
            x0_safe[2*N:] = np.maximum(x0_safe[2*N:], 0.0)
            
        try:
            # SLSQP handles bounds and constraints
            res = opt.minimize(objective, x0_safe, method='SLSQP', bounds=bounds, 
                               constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_res = res.x
        except Exception:
            pass

    # --- Initial Guess 1: 5x5 Grid + 1 Point ---
    # A 5x5 grid with radius 0.1 fits 25 circles perfectly.
    # We add one more circle in a gap and start with small radii to ensure feasibility.
    pts = []
    grid_coords = np.linspace(0.1, 0.9, 5)
    for x in grid_coords:
        for y in grid_coords:
            pts.append([x, y])
    # Add a point in a gap, e.g., (0.2, 0.2)
    pts.append([0.2, 0.2])
    
    centers1 = np.array(pts[:N])
    r1 = np.ones(N) * 0.05 # Small radius for feasible start
    x0_1 = np.concatenate([centers1[:, 0], centers1[:, 1], r1])
    try_optimize(x0_1, seed=42)

    # --- Initial Guess 2: Hexagonal-like Packing ---
    # Hexagonal packing is denser. We try to arrange 26 circles in rows.
    pts2 = []
    r_guess = 0.10
    dx = 2 * r_guess
    dy = np.sqrt(3) * r_guess
    # Row configurations summing to 26
    rows = [5, 6, 5, 6, 4] 
    current_y = 0.1
    shift = 0
    for count in rows:
        width = (count - 1) * dx
        start_x = (1.0 - width) / 2.0
        for k in range(count):
            x = start_x + k * dx + shift
            x = max(0.0, min(1.0, x))
            pts2.append([x, current_y])
        current_y += dy
        # Alternate shift for hexagonal pattern
        shift = r_guess if shift == 0 else 0
        
    centers2 = np.array(pts2[:N])
    centers2[:, 1] = np.clip(centers2[:, 1], 0.0, 1.0)
    r2 = np.ones(N) * 0.05
    x0_2 = np.concatenate([centers2[:, 0], centers2[:, 1], r2])
    try_optimize(x0_2, seed=123)

    # --- Initial Guess 3: Random ---
    np.random.seed(456)
    centers3 = np.random.rand(N, 2)
    # Keep points away from boundaries to help feasibility
    centers3 = np.clip(centers3, 0.1, 0.9)
    r3 = np.ones(N) * 0.04
    x0_3 = np.concatenate([centers3[:, 0], centers3[:, 1], r3])
    try_optimize(x0_3, seed=789)

    if best_res is not None:
        centers = np.column_stack([best_res[0:N], best_res[N:2*N]])
        radii = best_res[2*N:3*N]
        radii = np.maximum(radii, 0.0) # Ensure non-negative
        sum_r = np.sum(radii)
        return centers, radii, sum_r
    else:
        # Fallback solution
        centers = np.random.rand(N, 2)
        radii = np.ones(N) * 0.01
        return centers, radii, np.sum(radii)

# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state 9b0797fd) state=3394191f sum of radii=2.585555 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars, n):
    """Computes all inequality constraints as a single numpy array."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    c = []
    
    # Boundary constraints: circle must stay inside [0,1]x[0,1]
    c.extend(x - r)          # x >= r
    c.extend(1.0 - x - r)    # 1 - x >= r
    c.extend(y - r)          # y >= r
    c.extend(1.0 - y - r)    # 1 - y >= r
    
    # Overlap constraints: squared distance >= squared sum of radii
    for i in range(n):
        xi, yi, ri = x[i], y[i], r[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            dr = ri + r[j]
            c.append(dx*dx + dy*dy - dr*dr)
            
    return np.array(c, dtype=float)

def compute_objective(vars, n):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(vars[2::3])

def get_init_vars(n, method, seed):
    """Generates a feasible initial configuration."""
    np.random.seed(seed)
    vars = np.zeros(3 * n)
    
    if method == 'hex':
        # Hexagonal packing layout provides a dense, valid starting point
        r_init = 0.075
        y_step = r_init * np.sqrt(3)
        x_step = 2 * r_init
        pts = []
        row = 0
        col = 0
        while len(pts) < n:
            x = col * x_step + (row % 2) * r_init
            y = row * y_step
            if x <= 1.0 - r_init and y <= 1.0 - r_init:
                pts.append([x, y])
                col += 1
            else:
                row += 1
                col = 0
                
        for i in range(n):
            vars[3*i] = pts[i][0]
            vars[3*i+1] = pts[i][1]
            vars[3*i+2] = r_init
    else:
        # Random valid placement
        for i in range(n):
            r = np.random.uniform(0.05, 0.09)
            x = np.random.uniform(r, 1.0 - r)
            y = np.random.uniform(r, 1.0 - r)
            vars[3*i] = x
            vars[3*i+1] = y
            vars[3*i+2] = r
            
    # Add small noise to break symmetry and help escape local minima
    noise = np.random.normal(0, 1e-4, 3*n)
    vars = vars + noise
    
    # Project back to feasible region for safety
    for i in range(n):
        r = max(vars[3*i+2], 1e-5)
        vars[3*i] = np.clip(vars[3*i], r, 1.0 - r)
        vars[3*i+1] = np.clip(vars[3*i+1], r, 1.0 - r)
        vars[3*i+2] = r
    return vars

def constraint_func(vars):
    """Wrapper to match scipy signature without closures."""
    return compute_constraints(vars, 26)

def objective_func(vars):
    """Wrapper to match scipy signature without closures."""
    return compute_objective(vars, 26)

def run_packing():
    n = 26
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -1.0
    best_vars = None
    
    # Multiple restarts with different seeds and initialization strategies
    for seed in range(15):
        method = 'hex' if seed < 8 else 'random'
        x0 = get_init_vars(n, method, seed)
        
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_vars = res.x.copy()
        except Exception:
            pass
            
    # Fallback if optimization fails completely
    if best_vars is None:
        best_vars = get_init_vars(n, 'hex', 0)
        
    # Extract centers and radii
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = [best_vars[3*i], best_vars[3*i+1]]
        radii[i] = best_vars[3*i+2]
        
    # Validation and numerical repair
    valid = True
    for i in range(n):
        cx, cy, cr = centers[i,0], centers[i,1], radii[i]
        if cx - cr < -1e-7 or cx + cr > 1.0 + 1e-7 or cy - cr < -1e-7 or cy + cr > 1.0 + 1e-7:
            valid = False
        for j in range(i+1, n):
            dx = cx - centers[j,0]
            dy = cy - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - 1e-7:
                valid = False
                
    if not valid:
        # Slightly shrink radii until strictly valid
        for _ in range(50):
            radii *= 0.995
            is_valid = True
            for i in range(n):
                cx, cy, cr = centers[i,0], centers[i,1], radii[i]
                if cx - cr < 0 or cx + cr > 1.0 or cy - cr < 0 or cy + cr > 1.0:
                    is_valid = False
                    break
                for j in range(i+1, n):
                    dx = cx - centers[j,0]
                    dy = cy - centers[j,1]
                    if np.sqrt(dx*dx + dy*dy) < radii[i] + radii[j] - 1e-6:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                break

    return centers, radii, float(np.sum(radii))

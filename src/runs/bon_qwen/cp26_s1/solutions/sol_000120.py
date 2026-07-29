# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4ac25994) state=4c00f84c sum of radii=2.519536 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Return negative sum of radii for minimization."""
    return -np.sum(v[2::3])

def constraints(v):
    """Return array of inequality constraints (must be >= 0)."""
    n_constraints = 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    c = np.empty(n_constraints)
    idx = 0
    
    # Boundary constraints for each circle
    for i in range(N_CIRCLES):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        c[idx] = x - r;                idx += 1  # x >= r
        c[idx] = 1.0 - x - r;          idx += 1  # 1-x >= r
        c[idx] = y - r;                idx += 1  # y >= r
        c[idx] = 1.0 - y - r;          idx += 1  # 1-y >= r
        
    # Non-overlap constraints for each pair
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            ri_rj = v[3*i+2] + v[3*j+2]
            c[idx] = dist - ri_rj;      idx += 1
            
    return c

def generate_hex_init():
    """Hexagonal lattice initialization."""
    pts = []
    y = 0.15
    row = 0
    while len(pts) < N_CIRCLES:
        x = 0.15 + (row % 2) * 0.125
        while len(pts) < N_CIRCLES and x <= 0.9:
            pts.append((x, y))
            x += 0.25
        y += 0.22
        row += 1
    return np.array(pts[:N_CIRCLES])

def generate_grid_init():
    """Grid initialization."""
    pts = []
    for r in range(6):
        for c in range(5):
            if len(pts) >= N_CIRCLES:
                break
            pts.append((0.1 + c * 0.2, 0.1 + r * 0.2))
    return np.array(pts[:N_CIRCLES])

def generate_random_init():
    """Random initialization with fixed seed."""
    np.random.seed(42)
    return np.random.uniform(0.1, 0.9, (N_CIRCLES, 2))

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -1.0
    best_x = None
    
    # Prepare initial configurations
    init_configs = [generate_hex_init(), generate_grid_init(), generate_random_init()]
    
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    for cfg in init_configs:
        x0 = np.zeros(3 * N_CIRCLES)
        for i in range(N_CIRCLES):
            x0[3*i] = cfg[i, 0]
            x0[3*i+1] = cfg[i, 1]
            x0[3*i+2] = 0.09  # Initial radius guess
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Fallback to a trivial valid packing if optimization fails
    if best_x is None:
        centers = np.zeros((N_CIRCLES, 2))
        radii = np.zeros(N_CIRCLES)
        centers[0, 0] = 0.5
        centers[0, 1] = 0.5
        radii[0] = 0.4
        return centers, radii, 0.4
        
    # Extract results
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    # Apply negligible shrinkage to guarantee strict validity against numerical tolerance
    radii *= 0.9999999999
    
    return centers, radii, np.sum(radii)

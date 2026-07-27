# sol_000005 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=8af32522 sum of radii=2.622378 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(v):
    """Minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(v[2*N_CIRCLES:])

def constraint_func(v):
    """
    Returns an array of constraint values.
    Constraints are formulated as g(v) >= 0.
    Includes boundary constraints and pairwise non-overlap constraints.
    """
    centers = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = v[2*N_CIRCLES:]
    
    n_constraints = 4*N_CIRCLES + N_CIRCLES*(N_CIRCLES-1)//2
    con = np.empty(n_constraints)
    idx = 0
    
    # Boundary constraints
    for i in range(N_CIRCLES):
        con[idx] = centers[i,0] - radii[i]; idx += 1          # left
        con[idx] = 1.0 - centers[i,0] - radii[i]; idx += 1   # right
        con[idx] = centers[i,1] - radii[i]; idx += 1         # bottom
        con[idx] = 1.0 - centers[i,1] - radii[i]; idx += 1   # top
        
    # Non-overlap constraints
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            d = np.hypot(dx, dy)
            con[idx] = d - radii[i] - radii[j]
            idx += 1
            
    return con

def get_init_grid():
    """Generate a structured grid initialization with slight hexagonal offset."""
    v = np.zeros(3*N_CIRCLES)
    idx = 0
    cols, rows = 6, 5
    sx, sy = 1.0/(cols+1), 1.0/(rows+1)
    
    for r in range(rows):
        for c in range(cols):
            if idx < N_CIRCLES:
                # Apply half-step offset for odd rows to mimic hex packing
                cx = (c + 1) * sx + (r % 2) * 0.5 * sx
                cy = (r + 1) * sy
                v[2*idx] = np.clip(cx, 0.05, 0.95)
                v[2*idx+1] = np.clip(cy, 0.05, 0.95)
                v[2*N_CIRCLES+idx] = 0.08
                idx += 1
    return v

def get_init_random():
    """Generate a random initialization."""
    np.random.seed(42)
    centers = np.random.rand(2*N_CIRCLES) * 0.8 + 0.1
    radii = np.full(N_CIRCLES, 0.05)
    return np.concatenate([centers, radii])

def get_init_hex():
    """Generate a hexagonal-like initialization."""
    v = np.zeros(3*N_CIRCLES)
    idx = 0
    row_counts = [5, 5, 5, 5, 5, 1]
    y_pos = 0.1
    dy = 0.18
    
    for count in row_counts:
        x_start = (1.0 - count * 0.15) / 2.0
        for c in range(count):
            if idx < N_CIRCLES:
                v[2*idx] = x_start + c * 0.15
                v[2*idx+1] = y_pos
                v[2*N_CIRCLES+idx] = 0.07
                idx += 1
        y_pos += dy * np.sqrt(3)/2
    return v

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0) for _ in range(2*N_CIRCLES)] + [(1e-6, 0.5) for _ in range(N_CIRCLES)]
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    starts = [get_init_grid(), get_init_random(), get_init_hex()]
    best_val = -np.inf
    best_v = None
    
    # Run optimization from multiple starts
    for v0 in starts:
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False})
            
            val = -res.fun  # Sum of radii
            cons_val = constraint_func(res.x)
            
            # Accept if constraints are satisfied within tolerance and improves objective
            if np.min(cons_val) >= -1e-5 and val > best_val:
                best_val = val
                best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        best_v = starts[0].copy()
        
    centers = best_v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_v[2*N_CIRCLES:].copy()
    
    # Post-processing: Iteratively shrink radii to guarantee strict validity
    # This handles any residual numerical violations from the optimizer
    for _ in range(100):
        changed = False
        
        # Check pairwise overlaps
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-7:
                    shrink = (req - d) / 2.0 + 1e-8
                    radii[i] = max(radii[i] - shrink, 1e-6)
                    radii[j] = max(radii[j] - shrink, 1e-6)
                    changed = True
                    
        # Check boundary violations
        for i in range(N_CIRCLES):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-7:
                radii[i] = max(x, 1e-6)
                changed = True
            if x + r > 1.0 + 1e-7:
                radii[i] = max(1.0 - x, 1e-6)
                changed = True
            if y - r < -1e-7:
                radii[i] = max(y, 1e-6)
                changed = True
            if y + r > 1.0 + 1e-7:
                radii[i] = max(1.0 - y, 1e-6)
                changed = True
                
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))

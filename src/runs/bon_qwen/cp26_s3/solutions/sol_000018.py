# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0f0997f0) state=5bf86bbf sum of radii=2.623230 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def objective(vars, n):
    """Objective function: minimize negative sum of radii (equivalent to maximizing sum)."""
    r = vars[2 * n:]
    return -np.sum(r)

def constraint_func(vars, n):
    """
    Returns array of constraint values.
    All constraints must be >= 0 for SLSQP 'ineq' type.
    """
    c = vars[:2 * n].reshape(n, 2)
    r = vars[2 * n:]
    cons = []
    
    # Boundary constraints
    for i in range(n):
        cons.append(c[i, 0] - r[i])           # x >= r
        cons.append((1.0 - c[i, 0]) - r[i])   # 1-x >= r
        cons.append(c[i, 1] - r[i])           # y >= r
        cons.append((1.0 - c[i, 1]) - r[i])   # 1-y >= r
        
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            cons.append(dist - r[i] - r[j])   # dist >= r_i + r_j
            
    return np.array(cons)

def run_packing():
    n = 26
    
    # --- Initialization ---
    # Arrange circles in 5 rows with counts 5, 5, 5, 5, 6
    centers = np.zeros((n, 2))
    idx = 0
    y_coords = np.linspace(0.12, 0.88, 5)
    x_base = 0.12
    spacing = 0.22
    
    for row in range(5):
        y = y_coords[row]
        cols = 5 if row < 4 else 6
        # Offset odd rows for hexagonal packing
        shift = spacing / 2.0 if row % 2 == 1 else 0.0
        
        for col in range(cols):
            if idx >= n:
                break
            x = x_base + col * spacing + shift
            # Ensure initial positions are safely inside
            x = np.clip(x, 0.05, 0.95)
            centers[idx] = [x, y]
            idx += 1
            
    # Start with moderate radii to allow expansion during optimization
    radii = np.full(n, 0.06)
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Variable bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Setup constraints
    constraints = {
        'type': 'ineq',
        'fun': constraint_func,
        'args': (n,)
    }
    
    # --- Optimization ---
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        args=(n,),
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': 2000,
            'ftol': 1e-15,
            'disp': False
        }
    )
    
    final_centers = res.x[:2 * n].reshape(n, 2)
    final_radii = res.x[2 * n:]
    
    # Apply tiny shrink factor to guarantee strict validity with 1e-12 tolerance
    final_radii = final_radii * (1.0 - 1e-9)
    
    return final_centers, final_radii, np.sum(final_radii)

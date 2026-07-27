# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6773994b) state=35ec2037 sum of radii=2.604307 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Problem dimension and precomputed pairwise indices for non-overlap constraints
N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def compute_obj_and_cons(v):
    """Compute objective and constraints for the optimizer."""
    v = v.reshape(N, 3)
    x = v[:, 0]
    y = v[:, 1]
    r = v[:, 2]
    
    # Objective: maximize sum of radii -> minimize negative sum
    obj = -np.sum(r)
    
    # Inequality constraints (must be >= 0)
    c = []
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Non-overlap constraints: distance squared >= (r_i + r_j)^2
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    dist_sq = dx*dx + dy*dy
    sum_r = r[I_IDX] + r[J_IDX]
    c.append(dist_sq - sum_r**2)
    
    return obj, np.concatenate(c)

def objective(v):
    obj, _ = compute_obj_and_cons(v)
    return obj

def constraints(v):
    _, c = compute_obj_and_cons(v)
    return c

def get_initial_config():
    """Generate a structured hex-like initial layout with slight jitter."""
    centers = []
    # Row 0 (5 circles)
    for i in range(5):
        centers.append((0.15 + i*0.25, 0.1))
    # Row 1 (6 circles)
    for i in range(6):
        centers.append((0.225 + i*0.25, 0.3))
    # Row 2 (5 circles)
    for i in range(5):
        centers.append((0.15 + i*0.25, 0.5))
    # Row 3 (6 circles)
    for i in range(6):
        centers.append((0.225 + i*0.25, 0.7))
    # Row 4 (4 circles)
    for i in range(4):
        centers.append((0.225 + i*0.25, 0.9))
        
    centers = np.array(centers)
    # Add controlled random jitter to break symmetry
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    # Keep centers safely inside the square initially
    centers = np.clip(centers, 0.1, 0.9)
    
    # Start with small radii that strictly satisfy all constraints
    radii = np.full(N, 0.04)
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_val = -1.0
    best_centers = None
    best_radii = None
    
    # Multi-start optimization to escape local minima
    for seed in range(5):
        np.random.seed(100 + seed)
        centers, radii = get_initial_config()
        
        # Flatten to optimizer vector: [x0, y0, r0, x1, y1, r1, ...]
        v0 = np.zeros(N * 3)
        v0[0::3] = centers[:, 0]
        v0[1::3] = centers[:, 1]
        v0[2::3] = radii
        
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
        cons = {'type': 'ineq', 'fun': constraints}
        
        res = minimize(
            objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False}
        )
        
        if res.success:
            val = np.sum(res.x[2::3])
            if val > best_val:
                best_val = val
                best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                best_radii = res.x[2::3]
                
    # Fallback if optimization unexpectedly fails all starts
    if best_centers is None:
        return np.zeros((N, 2)), np.zeros(N), 0.0
        
    return best_centers, best_radii, best_val

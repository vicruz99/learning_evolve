import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2*N:])

def constraints(vars):
    """Constraint function: boundary and non-overlap constraints."""
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c_bnds = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Pairwise non-overlap constraints: dist - (r_i + r_j) >= 0
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, None] + radii[None, :]
    c_pairs = (dist - r_sum)[np.triu_indices(N, k=1)]
    
    return np.concatenate([c_bnds, c_pairs])

def get_initial(seed):
    """Generate initial configuration using a hexagonal grid with jitter."""
    rng = np.random.default_rng(seed)
    r0 = 0.08
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    while idx < N:
        y = r0 + row * np.sqrt(3) * r0
        x_start = r0 if row % 2 == 0 else 2 * r0
        x = x_start
        while x < 1.0 - r0 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r0
        row += 1
        
    # Add small random jitter to break symmetry
    centers += rng.uniform(-0.002, 0.002, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # Start with small radii to ensure initial feasibility
    radii = np.full(N, 0.04)
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_val = np.inf  # We minimize negative sum, so smaller is better
    
    # Run multiple trials to escape local optima
    for trial in range(10):
        init_vars = get_initial(trial * 100 + 42)
        try:
            res = minimize(objective, init_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
            if res.fun < best_val:
                best_val = res.fun
                best_vars = res.x
        except Exception:
            pass
            
    # Fallback if optimization fails (unlikely)
    if best_vars is None:
        best_vars = get_initial(0)
        best_vars[2*N:] = 0.05 
        
    centers = best_vars[:2*N].reshape(N, 2)
    radii = best_vars[2*N:]
    
    # Slight safety margin to guarantee constraint satisfaction within numerical tolerance
    # This prevents validation failures due to floating point inaccuracies
    radii = np.maximum(radii - 1e-10, 0.0)
    
    return centers, radii, np.sum(radii)
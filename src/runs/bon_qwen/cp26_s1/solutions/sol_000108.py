# sol_000108 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 81b841bb) state=2425ad77 sum of radii=2.620142 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    radii = vars[2 * N_CIRCLES:]
    return -np.sum(radii)

def get_constraints(vars):
    """Constraint function: returns array of values that must be >= 0."""
    n = N_CIRCLES
    centers = vars[:2 * n].reshape(n, 2)
    radii = vars[2 * n:]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    b1 = centers[:, 0] - radii
    b2 = 1.0 - centers[:, 0] - radii
    b3 = centers[:, 1] - radii
    b4 = 1.0 - centers[:, 1] - radii
    
    # Pairwise distance constraints: dist - (r1 + r2) >= 0
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract upper triangle indices to avoid duplicates and self-comparison
    idx = np.triu_indices(n, k=1)
    pairwise = dists[idx] - r_sums[idx]
    
    return np.concatenate([b1, b2, b3, b4, pairwise])

def generate_initial_guess(trial):
    """Generate feasible initial configurations."""
    n = N_CIRCLES
    r_init = 0.09
    centers = np.zeros((n, 2))
    idx = 0
    
    # Base layout: 5x5 grid
    for i in range(5):
        for j in range(5):
            if idx >= n:
                break
            centers[idx] = [0.1 + j * 0.2, 0.1 + i * 0.2]
            idx += 1
        if idx >= n:
            break
            
    # 26th circle at center
    if idx < n:
        centers[idx] = [0.5, 0.5]
        
    if trial == 0:
        # Standard grid
        pass
    elif trial == 1:
        # Slight shift
        centers[:, 0] += 0.02
        centers[:, 1] -= 0.02
    elif trial == 2:
        # Hex-like perturbation
        centers[:, 0] += 0.01 * np.random.randn(n)
        centers[:, 1] += 0.01 * np.random.randn(n)
    elif trial == 3:
        # Another random perturbation
        centers += 0.015 * np.random.uniform(-1, 1, centers.shape)
    elif trial == 4:
        # Different base scaling
        centers[:, 0] = 0.15 + (centers[:, 0] - 0.1) * 0.9
        centers[:, 1] = 0.15 + (centers[:, 1] - 0.1) * 0.9
        
    # Ensure strict initial feasibility
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(n, r_init)
    
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    np.random.seed(42)
    n = N_CIRCLES
    
    best_val = -np.inf
    best_vars = None
    
    # Bounds: x, y in [0, 1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    constraints = {'type': 'ineq', 'fun': get_constraints}
    
    # Run multiple optimizations from different starting points
    for trial in range(5):
        x0 = generate_initial_guess(trial)
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False}
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_vars = res.x
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_vars is None:
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.08)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= n: break
                centers[idx] = [0.15 + j * 0.17, 0.15 + i * 0.17]
                idx += 1
        if idx < n: centers[idx] = [0.5, 0.5]
        return centers, radii, np.sum(radii)
        
    centers = best_vars[:2 * n].reshape(n, 2)
    radii = best_vars[2 * n:]
    
    # Post-processing to guarantee strict validity
    # Clip to boundaries
    radii = np.clip(radii, 1e-6, 0.5)
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    # Iterative overlap resolution
    for _ in range(5):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                req = radii[i] + radii[j]
                if dist < req - 1e-9:
                    factor = dist / req
                    radii[i] *= np.sqrt(factor)
                    radii[j] *= np.sqrt(factor)
                    changed = True
        if not changed:
            break
            
    # Final boundary check
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        min_wall = min(x, 1.0 - x, y, 1.0 - y)
        if r > min_wall - 1e-9:
            radii[i] = min_wall
            
    return centers, radii, np.sum(radii)

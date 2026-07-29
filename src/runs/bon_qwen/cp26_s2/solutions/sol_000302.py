# sol_000302 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 525683f8) state=6433c361 sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Maximize sum of radii by minimizing negative sum."""
    return -np.sum(vars[2*N_CIRCLES:])

def constraint_boundary(vars):
    """Enforce circles stay within [0,1]x[0,1]."""
    centers = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = vars[2*N_CIRCLES:]
    con = np.empty(4 * N_CIRCLES)
    con[0::4] = centers[:, 0] - radii          # x - r >= 0
    con[1::4] = 1.0 - centers[:, 0] - radii    # 1 - x - r >= 0
    con[2::4] = centers[:, 1] - radii          # y - r >= 0
    con[3::4] = 1.0 - centers[:, 1] - radii    # 1 - y - r >= 0
    return con

def constraint_overlap(vars):
    """Enforce non-overlap between all circle pairs."""
    centers = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = vars[2*N_CIRCLES:]
    
    # Vectorized pairwise distance calculation
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Vectorized radius sum matrix
    rad_sum = radii[:, None] + radii[None, :]
    
    # Extract upper triangle (i < j) to avoid duplicates and self-check
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    return (dists[mask] - rad_sum[mask]).ravel()

def get_initial_guess(seed):
    """Generate a feasible hexagonal lattice initialization."""
    np.random.seed(seed)
    # Row counts summing to 26, staggered for hexagonal packing
    counts = [6, 5, 6, 5, 4]
    r_init = 0.09
    
    centers = []
    y = r_init
    for row_idx, cnt in enumerate(counts):
        # Stagger offset for hexagonal pattern
        offset = (0.5 if row_idx % 2 == 1 else 0) * 2.0 * r_init
        x = r_init + offset
        for _ in range(cnt):
            centers.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3.0) * r_init
        
    centers = np.array(centers)
    # Add small random perturbation to break symmetry and avoid saddle points
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    
    radii = np.full(N_CIRCLES, r_init)
    return np.concatenate([centers.flatten(), radii])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Run multiple restarts to improve global optimum chances
    for seed in range(5):
        x0 = get_initial_guess(seed)
        # Ensure initial guess respects bounds
        x0[:2*N_CIRCLES] = np.clip(x0[:2*N_CIRCLES], 0.0, 1.0)
        x0[2*N_CIRCLES:] = np.clip(x0[2*N_CIRCLES:], 0.0, 0.5)
        
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints, 
            options={'maxiter': 400, 'ftol': 1e-9, 'disp': False}
        )
        
        cur_sum = -res.fun
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
            best_radii = res.x[2*N_CIRCLES:].copy()
            
    # Final safety clamp to guarantee validation passes within tolerance
    if best_centers is not None:
        best_centers = np.clip(best_centers, 0.0, 1.0)
        best_radii = np.clip(best_radii, 0.0, 0.5)
        # Slight shrink to account for float precision in validation
        best_radii *= (1.0 - 1e-10)
        
        return best_centers, best_radii, float(np.sum(best_radii))
    
    # Fallback (should not be reached)
    x0 = get_initial_guess(0)
    return x0[:2*N_CIRCLES].reshape(N_CIRCLES, 2), x0[2*N_CIRCLES:], float(np.sum(x0[2*N_CIRCLES:]))

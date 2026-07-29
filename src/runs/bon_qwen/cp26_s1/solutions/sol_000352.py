# sol_000352 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b8d6b6a1) state=c4d0c545 sum of radii=2.605098 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_initial_hexagonal_packing(n, radius):
    """Generates initial centers for n circles in a hexagonal packing."""
    centers = []
    r = radius
    y = r
    while len(centers) < n:
        x = r
        shift = len(centers) % 2
        while x + r <= 1.0 + 1e-9:
            if len(centers) < n:
                centers.append([x, y])
            x += 2 * r + (r * shift)
        y += np.sqrt(3) * r
    return np.array(centers[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_sum = 0.0

    # Try multiple initial configurations to improve chances of finding a good local maximum
    for trial in range(3):
        # Initial radius guess (0.1 is typical for 25 circles, 0.09-0.1 for 26)
        r_guess = 0.09 + (trial * 0.002) 
        
        centers = generate_initial_hexagonal_packing(n, r_guess)
        radii = np.full(n, r_guess)
        
        # Add small random perturbation to avoid symmetry
        perturbation = (np.random.rand(n, 2) - 0.5) * 0.01
        centers = centers + perturbation
        centers = np.clip(centers, r_guess + 1e-5, 1.0 - r_guess - 1e-5)

        # Flattened variables: [x1..x26, y1..y26, r1..r26]
        x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n

        # Precompute indices for pairwise constraints
        triu_idx = np.triu_indices(n, k=1)

        def objective(v):
            return -np.sum(v[2*n:])

        def constraint_fun(v):
            x = v[:n]
            y = v[n:2*n]
            r = v[2*n:]
            c = np.column_stack((x, y))
            
            # Boundary constraints: c - r >= 0 and 1 - c - r >= 0
            bound_constraints = np.concatenate([
                x - r,
                1.0 - x - r,
                y - r,
                1.0 - y - r
            ])
            
            # Pairwise distance constraints
            diff = c[:, None, :] - c[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            pair_dists = dists[triu_idx]
            pair_r_sums = r[triu_idx[0]] + r[triu_idx[1]]
            
            return np.concatenate([bound_constraints, pair_dists - pair_r_sums])

        cons = {'type': 'ineq', 'fun': constraint_fun}

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success and res.fun < -best_sum:
                v_opt = res.x
                best_centers = np.column_stack((v_opt[:n], v_opt[n:2*n]))
                best_radii = v_opt[2*n:]
                best_sum = -res.fun
        except Exception:
            pass

    # Final validation and return
    if best_sum <= 0:
        # Fallback to a simple grid if optimization fails
        grid = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)][:26])
        best_centers = grid
        best_radii = np.full(26, 0.1)
        best_sum = 2.6

    return best_centers, best_radii, best_sum

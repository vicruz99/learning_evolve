# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0f0997f0) state=60a518e5 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)

    def objective(variables):
        radii = variables[2 * n:]
        return -np.sum(radii)

    def boundary_constraints(variables):
        centers = variables[:2 * n].reshape((n, 2))
        radii = variables[2 * n:]
        cons = []
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
            cons.append(x - r)
            cons.append(y - r)
            cons.append(1 - (x + r))
            cons.append(1 - (y + r))
        return np.array(cons)

    def overlap_constraints(variables):
        centers = variables[:2 * n].reshape((n, 2))
        radii = variables[2 * n:]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                cons.append(dist - radii[i] - radii[j])
        return np.array(cons)

    def generate_hexagonal_seed(r_approx=0.09):
        centers = []
        r = r_approx
        # Hexagonal rows
        row_y = r
        while row_y + r <= 1.0:
            row_x = r
            offset = (len(centers) % 2) * r
            if offset > 0:
                row_x = r + offset
            while row_x + r <= 1.0 and len(centers) < n:
                centers.append([row_x, row_y])
                row_x += 2 * r
            row_y += r * np.sqrt(3)
        return np.array(centers[:n])

    # Run optimization with multiple restarts to avoid local minima
    for restart in range(3):
        if restart == 0:
            # Hexagonal initialization
            centers_init = generate_hexagonal_seed()
            # Add slight jitter
            centers_init += np.random.uniform(-0.001, 0.001, centers_init.shape)
        else:
            # Random initialization with safe bounds
            centers_init = np.random.uniform(0.15, 0.85, (n, 2))
        
        radii_init = np.full(n, 0.08)
        
        initial_state = np.concatenate([centers_init.flatten(), radii_init])

        # Constraints for scipy
        bounds = []
        for _ in range(2 * n):
            bounds.append((0, 1)) # x, y in [0, 1]
        for _ in range(n):
            bounds.append((0, 0.5)) # r in [0, 0.5]

        bnds = tuple(bounds)
        
        # Constraint dicts
        con_boundary = {'type': 'ineq', 'fun': boundary_constraints}
        con_overlap = {'type': 'ineq', 'fun': overlap_constraints}

        try:
            result = minimize(
                objective, 
                initial_state, 
                method='SLSQP', 
                bounds=bnds, 
                constraints=[con_boundary, con_overlap],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < -best_sum:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = result.x[:2 * n].reshape((n, 2))
                    best_radii = result.x[2 * n:]
        except Exception:
            continue

    # Final Validation Check
    centers = best_centers
    radii = best_radii
    
    # Clip values slightly to ensure strict validity if on the edge
    # This handles tiny numerical errors from the optimizer
    radii = np.maximum(radii, 0.0)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        centers[i, 0] = np.clip(x, r, 1 - r)
        centers[i, 1] = np.clip(y, r, 1 - r)
        
    return centers, radii, best_sum

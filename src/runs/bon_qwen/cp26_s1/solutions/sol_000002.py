# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=5957c717 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_hexagonal_packing(n=26):
    """
    Generates an initial configuration of circles in a hexagonal lattice pattern.
    """
    r_initial = 0.09
    centers = []
    row = 0
    while len(centers) < n:
        y = r_initial + row * (r_initial * np.sqrt(3))
        if y + r_initial > 1.0:
            break
        
        # Shift odd rows horizontally
        x_start = 2 * r_initial if row % 2 == 1 else r_initial
        x = x_start
        while x + r_initial <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_initial
        row += 1
        
    centers = np.array(centers[:n])
    radii = np.full(n, r_initial)
    return centers, radii

def check_validity(centers, radii):
    """
    Checks validity constraints for the optimization process.
    """
    n = len(radii)
    if np.any(radii < 0):
        return False
    if np.any(centers < 0) or np.any(centers > 1.0):
        return False
    
    # Boundary check
    for i in range(n):
        if (centers[i, 0] - radii[i] < -1e-9) or (centers[i, 0] + radii[i] > 1.0 + 1e-9):
            return False
        if (centers[i, 1] - radii[i] < -1e-9) or (centers[i, 1] + radii[i] > 1.0 + 1e-9):
            return False
            
    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Initialization ---
    centers, radii = generate_hexagonal_packing(n)
    var_init = np.concatenate([centers.flatten(), radii])
    
    # --- Optimization Objective ---
    def objective(vars_flat):
        radii_vars = vars_flat[-n:]
        # We want to maximize sum(radii), so we minimize negative sum
        return -np.sum(radii_vars)
    
    # --- Constraints ---
    constraints = []
    bounds = []

    for i in range(n):
        # Bounds for centers (0 to 1) and radii (0 to 0.5)
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        ix, iy = 3 * i, 3 * i + 1
        ir = 3 * i + 2
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[ix] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[ix] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[iy] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[iy] - v[ir]})

    for i in range(n):
        for j in range(i + 1, n):
            ix, iy = 3 * i, 3 * i + 1
            ir_i = 3 * i + 2
            jx, jy = 3 * j, 3 * j + 1
            ir_j = 3 * j + 2
            
            # Constraint: dist^2 >= (r_i + r_j)^2
            def make_constraint(i_idx, j_idx):
                def constraint(v):
                    dx = v[3 * i_idx] - v[3 * j_idx]
                    dy = v[3 * i_idx + 1] - v[3 * j_idx + 1]
                    dist_sq = dx**2 + dy**2
                    rad_sum_sq = (v[3 * i_idx + 2] + v[3 * j_idx + 2])**2
                    return dist_sq - rad_sum_sq
                return constraint

            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # --- Multi-Stage Optimization ---
    best_sum = -1.0
    best_state = var_init

    # Stage 1: Main optimization from hexagonal start
    res = minimize(objective, var_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000})
    if res.success and check_validity(res.x[:-n].reshape(n, 2), res.x[-n:]):
        if -res.fun > best_sum:
            best_sum = -res.fun
            best_state = res.x

    # Stage 2: Perturbation and refinement to escape local minima
    for _ in range(5):
        perturb = np.random.normal(0, 0.002, size=var_init.shape)
        perturbed_state = best_state + perturb
        # Ensure radii stay non-negative after perturbation
        perturbed_state[3::3] = np.maximum(perturbed_state[3::3], 0.01)
        
        res2 = minimize(objective, perturbed_state, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        if res2.success and check_validity(res2.x[:-n].reshape(n, 2), res2.x[-n:]):
            if -res2.fun > best_sum:
                best_sum = -res2.fun
                best_state = res2.x

    # --- Final Formatting ---
    final_centers = best_state[:-n].reshape(n, 2)
    final_radii = best_state[-n:]
    
    # Final validation check
    if not check_validity(final_centers, final_radii):
        # Fallback to initial if optimization somehow failed validation
        return centers, radii, np.sum(radii)

    return final_centers, final_radii, float(np.sum(final_radii))

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")

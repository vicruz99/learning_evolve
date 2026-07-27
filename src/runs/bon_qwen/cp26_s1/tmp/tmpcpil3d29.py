import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def zero_obj(x):
    """Dummy objective function for SLSQP when maximizing radius externally."""
    return 0.0

def compute_constraints(x_flat, r):
    """
    Computes boundary and non-overlap constraints.
    Returns an array where each element must be >= 0 for feasibility.
    """
    centers = x_flat.reshape(-1, 2)
    n = centers.shape[0]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    b_cons = np.empty(n * 4)
    b_cons[0::4] = centers[:, 0] - r
    b_cons[1::4] = 1.0 - centers[:, 0] - r
    b_cons[2::4] = centers[:, 1] - r
    b_cons[3::4] = 1.0 - centers[:, 1] - r

    # Overlap constraints: ||c_i - c_j|| - 2r >= 0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist_pairs = dist[mask]
    o_cons = dist_pairs - 2.0 * r

    return np.concatenate([b_cons, o_cons])

def generate_initial_layout(restart_idx):
    """Generates a hexagonal-ish initial layout with slight deterministic perturbation."""
    np.random.seed(42 + restart_idx)
    pos = []
    # Base hex grid parameters
    dx, dy = 0.18, 0.155
    offset_x = 0.12
    offset_y = 0.12
    
    for i in range(7):
        for j in range(6):
            if len(pos) >= N_CIRCLES:
                break
            x = offset_x + j * dx + (dx / 2.0 if i % 2 == 1 else 0.0)
            y = offset_y + i * dy
            if x <= 0.95 and y <= 0.95:
                # Add small controlled noise to break symmetry
                x += np.random.uniform(-0.005, 0.005)
                y += np.random.uniform(-0.005, 0.005)
                pos.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)])
                
    # Fallback if grid generation falls short (shouldn't happen with params above)
    while len(pos) < N_CIRCLES:
        pos.append(np.random.uniform(0.1, 0.9, 2))
        
    return np.array(pos[:N_CIRCLES]).flatten()

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES)
    best_r = 0.0
    best_x = None
    
    # Run multiple restarts to escape local minima
    for restart in range(5):
        x_curr = generate_initial_layout(restart)
        r = 0.045  # Start safely feasible
        
        # Iteratively grow radius
        for _ in range(180):
            cons_dict = {
                'type': 'ineq',
                'fun': compute_constraints,
                'args': (r,)
            }
            
            res = minimize(
                fun=zero_obj,
                x0=x_curr,
                method='SLSQP',
                bounds=bounds,
                constraints=cons_dict,
                options={'maxiter': 40, 'ftol': 1e-11, 'disp': False}
            )
            
            # Verify feasibility of the result
            c_vals = compute_constraints(res.x, r)
            if np.min(c_vals) >= -1e-6:
                x_curr = res.x
                best_r = r
                best_x = res.x.copy()
                r *= 1.0065  # Increase radius
            else:
                # Infeasible: shrink radius slightly and continue or break
                r *= 0.995
                if r < 0.04:
                    break
                    
        # Final push: try to increase r by small increments until failure
        while True:
            r_test = best_r * 1.003
            cons_test = {
                'type': 'ineq',
                'fun': compute_constraints,
                'args': (r_test,)
            }
            res_test = minimize(
                fun=zero_obj,
                x0=best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=cons_test,
                options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
            )
            if np.min(compute_constraints(res_test.x, r_test)) >= -1e-6:
                best_r = r_test
                best_x = res_test.x.copy()
            else:
                break

    # Ensure final configuration is strictly valid
    centers = best_x.reshape(-1, 2)
    # Clamp to safe bounds just in case of numerical drift
    centers[:, 0] = np.clip(centers[:, 0], best_r, 1.0 - best_r)
    centers[:, 1] = np.clip(centers[:, 1], best_r, 1.0 - best_r)
    
    radii = np.full(N_CIRCLES, best_r)
    return centers, radii, np.sum(radii)
# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=1842376a sum of radii=2.610265 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def compute_constraints(vars_vec):
    """
    Constraint function: pairwise non-overlap constraints >= 0.
    Boundary constraints are handled automatically by the u, v parameterization.
    """
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]

    # Transform normalized coordinates to actual positions within [r, 1-r]
    # u=0 -> x=r (touching left wall), u=1 -> x=1-r (touching right wall)
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)

    # Pairwise squared distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist_sq = dx*dx + dy*dy

    # Required squared distances for non-overlap
    r_sum = r[i_idx] + r[j_idx]
    min_dist_sq = r_sum * r_sum

    return dist_sq - min_dist_sq

def generate_initial_vars(seed):
    """Generates initial variables for optimization."""
    np.random.seed(seed)
    vars0 = np.empty(3 * N)

    # Initialize radii to a safe value that guarantees initial feasibility
    r_init = 0.04 + 0.015 * np.random.rand()
    vars0[0::3] = r_init

    # Generate hexagonal lattice in normalized u,v space
    u_vals = []
    v_vals = []
    y_idx = 0
    # Row structure summing to 26
    row_counts = [6, 5, 6, 5, 4]  

    # Spacing tuned for normalized [0,1] box
    u_step = 0.20
    v_step = 0.20

    for r_idx, count in enumerate(row_counts):
        # Shift odd rows for hexagonal packing
        shift = 0.10 if r_idx % 2 == 1 else 0.0
        for c in range(count):
            u_vals.append(0.05 + shift + c * u_step)
            v_vals.append(0.05 + y_idx * v_step)
        y_idx += 1

    u_vals = np.array(u_vals[:N])
    v_vals = np.array(v_vals[:N])

    # Add controlled perturbation to escape symmetry and local minima
    u_vals += np.random.uniform(-0.03, 0.03, N)
    v_vals += np.random.uniform(-0.03, 0.03, N)

    # Clip to valid normalized range
    u_vals = np.clip(u_vals, 0.0, 1.0)
    v_vals = np.clip(v_vals, 0.0, 1.0)

    vars0[1::3] = u_vals
    vars0[2::3] = v_vals

    return vars0

def run_packing():
    best_vars = None
    best_val = -np.inf

    # Bounds: r in [1e-6, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}

    # Phase 1: Multiple restarts from diverse initializations
    num_restarts = 30
    for seed in range(num_restarts):
        vars0 = generate_initial_vars(seed)

        try:
            res = minimize(compute_objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'iprint': -1})

            # Check constraint satisfaction within numerical tolerance
            con_vals = compute_constraints(res.x)
            if np.min(con_vals) >= -1e-6:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_vars = res.x.copy()
        except Exception:
            continue

    # Phase 2: Local perturbation refinement to escape local minima
    if best_vars is not None:
        np.random.seed(42)
        for _ in range(20):
            x_pert = best_vars.copy()
            # Perturb radii and normalized positions slightly
            x_pert[0::3] += np.random.uniform(-0.001, 0.001, N)
            x_pert[1::3] += np.random.uniform(-0.01, 0.01, N)
            x_pert[2::3] += np.random.uniform(-0.01, 0.01, N)

            # Enforce bounds strictly
            x_pert[0::3] = np.clip(x_pert[0::3], 1e-6, 0.5)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.0, 1.0)
            x_pert[2::3] = np.clip(x_pert[2::3], 0.0, 1.0)

            try:
                res = minimize(compute_objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'iprint': -1})

                con_vals = compute_constraints(res.x)
                if np.min(con_vals) >= -1e-6:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_vars = res.x.copy()
            except Exception:
                continue

        # Phase 3: High-precision polish on the best configuration
        if best_vars is not None:
            try:
                res_final = minimize(compute_objective, best_vars, method='SLSQP', bounds=bounds,
                                     constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'iprint': -1})
                if np.min(compute_constraints(res_final.x)) >= -1e-6:
                    best_vars = res_final.x
            except Exception:
                pass

    # Fallback to a valid configuration if optimization fails completely
    if best_vars is None:
        r_fall = 0.04
        u_fall = np.linspace(0.0, 1.0, N)
        v_fall = np.linspace(0.0, 1.0, N)
        best_vars = np.empty(3 * N)
        best_vars[0::3] = r_fall
        best_vars[1::3] = u_fall
        best_vars[2::3] = v_fall
        best_val = N * r_fall

    # Reconstruct centers from optimized parameters
    radii = best_vars[0::3]
    u = best_vars[1::3]
    v = best_vars[2::3]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    centers = np.column_stack((x, y))

    # Ensure non-negative radii against numerical drift
    radii = np.maximum(radii, 0.0)

    return centers, radii, float(np.sum(radii))

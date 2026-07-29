# sol_000365 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b4024b4) state=4c33de06 sum of radii=2.620922 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, differential_evolution

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_sum = 0.0
    best_state = None

    # Objective function: minimize negative sum of radii
    def objective(x):
        # x contains 26 radii followed by 52 coordinates
        radii = x[:n]
        centers = x[n:].reshape((n, 2))
        return -np.sum(radii)

    # Helper for constraints
    def boundary_and_overlap(x):
        radii = x[:n]
        centers = x[n:].reshape((n, 2))
        constraints_list = []

        # Boundary constraints: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        for i in range(n):
            constraints_list.append(centers[i, 0] - radii[i])
            constraints_list.append(1 - (centers[i, 0] + radii[i]))
            constraints_list.append(centers[i, 1] - radii[i])
            constraints_list.append(1 - (centers[i, 1] + radii[i]))
            constraints_list.append(radii[i]) # Non-negative radii

        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                constraints_list.append(dist - radii[i] - radii[j])
        
        return constraints_list

    def get_constraints_dict(x):
        c = boundary_and_overlap(x)
        constraints = [{'type': 'ineq', 'fun': lambda x, idx=idx: boundary_and_overlap(x)[idx]} for idx in range(len(c))]
        return constraints

    # Strategy: Multiple restarts with SLSQP
    for attempt in range(10):
        # Generate a good initial guess
        # 1. Hexagonal lattice pattern (5-4-5-4-5-3)
        radii = np.full(n, 0.08)
        centers = np.zeros((n, 2))
        
        rows = [5, 4, 5, 4, 5, 3]
        idx = 0
        for r_idx, count in enumerate(rows):
            y = 0.5 + (r_idx - 2.5) * 0.15 # Staggered y
            if r_idx % 2 != 0:
                y += 0.075 # Shift staggered rows
                x_start = 0.5 - (count - 1) * 0.1
            else:
                x_start = 0.5 - (count - 1) * 0.1
            
            for k in range(count):
                x = x_start + k * 0.2
                centers[idx, 0] = x
                centers[idx, 1] = y
                idx += 1
        
        # Randomize slightly to break symmetry
        centers += np.random.uniform(-0.02, 0.02, (n, 2))
        
        # Combine into state vector
        x0 = np.concatenate([radii, centers.flatten()])
        
        # Bounds: radii in [0, 0.5], coords in [0, 1]
        bounds = [(0, 0.5)] * n + [(0, 1)] * (2 * n)

        try:
            # Use SLSQP for local optimization
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=[{'type': 'ineq', 'fun': lambda x: boundary_and_overlap(x)}],
                           options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success or -res.fun > best_sum:
                # Post-process: ensure strict validity by slight shrinking if needed
                curr_radii = res.x[:n]
                curr_centers = res.x[n:].reshape((n, 2))
                curr_sum = np.sum(curr_radii)
                
                # Validate and fix numerical drift
                if validate_packing(curr_centers, curr_radii):
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_state = (curr_centers, curr_radii)
        except Exception:
            continue

    # If we haven't found a good packing, fallback to a simple valid one
    if best_state is None:
        best_radii = np.full(n, 0.05)
        best_centers = np.array([[i%5*0.2+0.1, i//5*0.2+0.1] for i in range(n)])
        best_sum = np.sum(best_radii)

    # Final check and return
    final_centers, final_radii = best_state
    if not validate_packing(final_centers, final_radii):
        # Emergency shrink
        while not validate_packing(final_centers, final_radii):
            final_radii *= 0.99
            
    return final_centers, final_radii, np.sum(final_radii)

def validate_packing(centers, radii):
    """
    Validation function (Read-Only)
    """
    import numpy as np
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

# Ensure validate_packing is accessible at top level if not already defined globally
# (The prompt provided it, but we define a local one for self-contained logic if needed, 
# though the prompt says "do not modify", implying it exists in the environment. 
# We will assume the environment has it or we use our own copy for safety.)

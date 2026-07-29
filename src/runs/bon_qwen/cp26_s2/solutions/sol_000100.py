# sol_000100 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2a23e4d6) state=e72e830b sum of radii=2.522844 correctness=1.0
# stdout(first 200): Running optimization restart 1/10... New best sum found: 2.489975 Running optimization restart 2/10... Running optimization restart 3/10... Running optimization restart 4/10... Running optimization re
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

def get_constraints(centers, radii):
    """
    Computes the constraint values for the packing problem.
    Returns an array where all values must be >= 0.
    Includes boundary constraints and non-overlap constraints.
    """
    n = centers.shape[0]
    constraints = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # x - r >= 0  => constraint value: x - r
    constraints.append(centers[:, 0] - radii)
    constraints.append(1 - centers[:, 0] - radii)
    constraints.append(centers[:, 1] - radii)
    constraints.append(1 - centers[:, 1] - radii)
    
    # Non-overlap constraints: dist(i, j) >= r_i + r_j
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # We use the squared form to avoid square roots, but note that 
    # a^2 >= b^2 is equivalent to |a| >= |b| for non-negative a, b.
    # Here dist >= 0 and sum_radii >= 0, so it holds.
    # However, for numerical stability, sometimes direct distance is preferred.
    # Let's use direct distance: dist - (r_i + r_j) >= 0.
    
    # Vectorized pairwise distance calculation
    # centers shape (n, 2)
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # radii sum matrix shape (n, n)
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # We only need upper triangle (i < j)
    # dists[i, j] - rad_sum[i, j] >= 0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap_vals = dists[mask] - rad_sum[mask]
    constraints.append(overlap_vals)
    
    return np.concatenate(constraints)

def objective_and_constraints(vars_flat, n):
    """
    Wrapper for the optimizer.
    """
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    
    obj = -np.sum(radii)
    cons = get_constraints(centers, radii)
    
    return obj, cons

def objective(vars_flat, n):
    radii = vars_flat[2*n:]
    return -np.sum(radii)

def constraint_fun(vars_flat, n):
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    return get_constraints(centers, radii)

def generate_grid_initialization(n, seed=None):
    """Generates a grid-like initialization for n circles."""
    if seed is not None:
        np.random.seed(seed)
    
    # Try to fit n circles in a grid
    # Estimate rows and cols
    aspect = 1.0
    cols = int(np.ceil(np.sqrt(n / aspect)))
    rows = int(np.ceil(n / cols))
    
    # Adjust to make it more square if needed
    while (cols - 1) * (rows) > n + 2: # heuristic
        cols -= 1
        rows = int(np.ceil(n / cols))
    
    points = []
    for r in range(rows):
        for c in range(cols):
            if len(points) < n:
                # Normalize to [0, 1] with padding
                x = (c + 0.5) / cols
                y = (r + 0.5) / rows
                points.append([x, y])
            else:
                break
    
    centers = np.array(points[:n])
    # Add some noise
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(n, 0.01)
    return centers, radii

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
    
    # Define constraints for SLSQP
    # We need a function that returns the constraint values
    # SLSQP handles inequality constraints g(x) >= 0
    
    def cons_func(vars_flat):
        return constraint_fun(vars_flat, n)

    constraint_dict = {
        'type': 'ineq',
        'fun': cons_func
    }
    
    # Try multiple restarts
    num_restarts = 10
    
    for i in range(num_restarts):
        print(f"Running optimization restart {i+1}/{num_restarts}...")
        
        # Generate initial guess
        centers, radii = generate_grid_initialization(n, seed=i)
        
        # Flatten variables
        x0 = np.concatenate([centers.flatten(), radii])
        
        try:
            # Run optimizer
            result = minimize(
                fun=objective,
                x0=x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint_dict],
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = result.x[:2*n].reshape(n, 2)
                    best_radii = result.x[2*n:]
                    print(f"New best sum found: {current_sum:.6f}")
            else:
                # Even if not successful, check if it's better
                # Sometimes it stops early but is valid
                centers_temp = result.x[:2*n].reshape(n, 2)
                radii_temp = result.x[2*n:]
                
                # Check validity roughly
                try:
                    cons_val = get_constraints(centers_temp, radii_temp)
                    if np.min(cons_val) >= -1e-6: # Allow tiny numerical error
                        current_sum = -np.sum(radii_temp)
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_centers = centers_temp
                            best_radii = radii_temp
                            print(f"Valid solution found (not converged): {current_sum:.6f}")
                except:
                    pass

        except Exception as e:
            print(f"Optimization failed: {e}")
            continue

    # Final refinement: ensure strict validity by shrinking slightly if needed
    if best_centers is not None:
        # Check for any tiny violations and shrink radii to fix them
        # This is a safeguard against numerical noise
        violations = get_constraints(best_centers, best_radii)
        min_violation = np.min(violations)
        
        if min_violation < 0:
            # There is a violation. We need to shrink radii.
            # A simple global shrink might work, or just re-run with fixed constraints?
            # Given the target, numerical noise is likely the culprit if close.
            # Let's shrink radii by a small epsilon to clear violations
            # Find max violation magnitude
            max_v = -min_violation
            if max_v > 1e-6:
                # Estimate how much to shrink. 
                # Violation in dist constraint: dist < r_i + r_j => violation = r_i + r_j - dist
                # Shrinking both by delta reduces sum by 2*delta.
                # Violation in boundary: r > x => violation = r - x. Shrinking r by delta fixes it.
                # So delta = max_v should be safe? 
                # Actually for dist constraint, reducing r_i and r_j by max_v/2 each?
                # Let's just shrink all radii by max_v + epsilon
                shrink = max_v + 1e-7
                best_radii = np.maximum(best_radii - shrink, 0)
                
        # Clip centers to [0, 1] just in case
        best_centers = np.clip(best_centers, 0.0, 1.0)
        
        # Recalculate sum
        best_sum = np.sum(best_radii)
        print(f"Final validated sum: {best_sum:.6f}")

    return best_centers, best_radii, best_sum

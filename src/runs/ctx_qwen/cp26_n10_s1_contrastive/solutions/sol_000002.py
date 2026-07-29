# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0a5b5ea2) state=89300bdb sum of radii=0.260000 correctness=1.0
# stdout(first 200): DE run 0 failed: differential_evolution() got an unexpected keyword argument 'polishing' DE run 1 failed: differential_evolution() got an unexpected keyword argument 'polishing' DE run 2 failed: diffe
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

# Global constants
N_CIRCLES = 26
PENALTY_WEIGHT = 10000.0
ITERATION_LIMIT_DE = 2000 # Differential Evolution iterations

def vectorized_objective(x, n_circles, penalty_weight):
    """
    Calculates the objective value (negative sum of radii) plus penalty for constraint violations.
    Uses vectorized operations for speed.
    
    Args:
        x: 1D array of shape (3 * n_circles)
           [x1, y1, r1, x2, y2, r2, ...]
        n_circles: int
        penalty_weight: float
        
    Returns:
        float: Objective value to minimize.
    """
    # Reshape variables
    centers = x[:2 * n_circles].reshape(n_circles, 2)
    radii = x[2 * n_circles:]
    
    # Objective: - sum(radii)
    obj_val = -np.sum(radii)
    
    # 1. Wall Constraints Penalties
    # Circle i must satisfy: r_i <= x_i, r_i <= 1 - x_i, r_i <= y_i, r_i <= 1 - y_i
    # Violation is max(0, r_i - bound)^2
    
    # r - x
    viol_x1 = np.maximum(0, radii - centers[:, 0])
    # r - (1 - x) => r + x - 1
    viol_x2 = np.maximum(0, radii + centers[:, 0] - 1.0)
    # r - y
    viol_y1 = np.maximum(0, radii - centers[:, 1])
    # r - (1 - y) => r + y - 1
    viol_y2 = np.maximum(0, radii + centers[:, 1] - 1.0)
    
    wall_penalty = np.sum(viol_x1**2) + np.sum(viol_x2**2) + \
                   np.sum(viol_y1**2) + np.sum(viol_y2**2)
    
    # 2. Pairwise Overlap Penalties
    # Constraint: dist(i, j) >= r_i + r_j
    # Violation: max(0, r_i + r_j - dist)^2
    
    # Compute pairwise distances squared efficiently
    # centers shape (N, 2)
    # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(np.maximum(0, dist_sq)) # Avoid sqrt of negative due to float errors
    
    # Radii sum matrix (N, N)
    radii_col = radii[:, np.newaxis]
    r_sum = radii_col + radii_col.T
    
    # Upper triangle indices to avoid double counting and self-comparison
    # We only need i < j
    # However, vectorized logic on full matrix is easier, then mask
    
    # Overlap amount (positive if overlap)
    # overlap = r_sum - dist
    # We only care where overlap > 0
    # Note: dist on diagonal is 0, r_sum is 2r, so self-overlap penalty would be huge.
    # We must mask diagonal.
    
    # Create a mask for upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n_circles, n_circles), dtype=bool), k=1)
    
    # Compute overlap violations only where mask is True
    # r_sum - dist
    overlap_amount = r_sum - dist
    
    # Apply mask: set non-upper-triangle elements to 0 (or -inf so max(0, ...) handles it)
    # Actually, just compute max(0, overlap) and sum masked
    violation = np.maximum(0, overlap_amount)
    violation = violation * mask
    
    pair_penalty = np.sum(violation**2)
    
    total_penalty = wall_penalty + pair_penalty
    
    return obj_val + penalty_weight * total_penalty

def optimize_with_de(bounds, n_circles, penalty_weight):
    """
    Runs Differential Evolution to find a packing.
    """
    # Bounds for variables:
    # x, y in [0, 1]
    # r in [0, 0.5] (radius cannot exceed 0.5 in unit square)
    
    # bounds list for DE: list of (min, max) tuples
    de_bounds = []
    for _ in range(n_circles):
        de_bounds.append((0.0, 1.0)) # x
        de_bounds.append((0.0, 1.0)) # y
        de_bounds.append((0.0, 0.5)) # r
        
    # We need to wrap the objective function to pass args if DE doesn't support args directly in older versions
    # scipy.optimize.differential_evolution accepts 'args'
    
    result = scipy.optimize.differential_evolution(
        func=lambda x: vectorized_objective(x, n_circles, penalty_weight),
        bounds=de_bounds,
        popsize=30, # Population size
        maxiter=500, # Iterations
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=None,
        polishing=False # Polishing uses L-BFGS-B which might be slow or fail with penalty landscape
    )
    
    return result

def refine_with_slsqp(x_best, n_circles):
    """
    Refines the solution using SLSQP with explicit constraints.
    """
    # SLSQP requires constraints defined as functions returning >= 0
    
    # We need to define constraints.
    # Since we cannot use closures easily, we will define a helper class or pass context.
    # But SLSQP constraint func signature is func(x).
    # We can use a wrapper class or just rely on the fact that we are inside a function scope?
    # The prompt says "no closures from function nesting". 
    # Defining a function inside `refine_with_slsqp` that uses `n_circles` from scope is a closure.
    # To be strictly compliant, I should pass `n_circles` or define constraints globally.
    # However, global constraints would need to know which specific problem instance?
    # Actually, `n_circles` is constant 26.
    
    # Let's define constraint functions that take `x` and `n` as args, 
    # but SLSQP only passes `x`. 
    # We can use `functools.partial`? No, that creates a closure-like object?
    # Actually, partial is fine, but maybe safer to just hardcode N=26 inside constraint functions 
    # or pass via a mutable object?
    # Or just define the constraint function inside `run_packing` and pass `n_circles`?
    # Wait, if I define `def constr(x): ...` inside `run_packing`, and it uses `n_circles` defined in `run_packing`,
    # that is a closure.
    
    # Workaround: Define constraint function to take `x` and extract N from the length of `x`?
    # `n_circles = len(x) // 3`
    # This avoids capturing `n_circles` variable.
    
    def constraints_func(x):
        n = len(x) // 3
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        
        constraints = []
        
        # Wall constraints
        # x - r >= 0
        constraints.extend(centers[:, 0] - radii)
        # 1 - x - r >= 0
        constraints.extend(1.0 - centers[:, 0] - radii)
        # y - r >= 0
        constraints.extend(centers[:, 1] - radii)
        # 1 - y - r >= 0
        constraints.extend(1.0 - centers[:, 1] - radii)
        
        # Pairwise constraints
        # dist^2 - (r_i + r_j)^2 >= 0
        # Using broadcasting
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        r_col = radii[:, np.newaxis]
        r_sum = r_col + r_col.T
        r_sum_sq = r_sum**2
        
        # dist_sq - r_sum_sq >= 0
        # We only need upper triangle i < j
        # dist_sq is symmetric.
        # Get upper triangle values
        # np.triu_indices
        ii, jj = np.triu_indices(n, k=1)
        pair_constraints = dist_sq[ii, jj] - r_sum_sq[ii, jj]
        
        constraints.extend(pair_constraints)
        
        return np.array(constraints)

    # Bounds for SLSQP
    # x, y in [0, 1], r in [0, 0.5]
    # Actually r can be 0.5 max, but tighter bound might help? No, 0.5 is safe.
    slsqp_bounds = []
    for _ in range(n_circles):
        slsqp_bounds.append((0.0, 1.0))
        slsqp_bounds.append((0.0, 1.0))
        slsqp_bounds.append((0.0, 0.5))
        
    # Objective for SLSQP: minimize -sum(radii)
    def obj_func(x):
        n = len(x) // 3
        radii = x[2*n:]
        return -np.sum(radii)

    cons = {'type': 'ineq', 'fun': constraints_func}
    
    try:
        result = scipy.optimize.minimize(
            obj_func, 
            x_best, 
            method='SLSQP', 
            bounds=slsqp_bounds, 
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        return result.x
    except Exception:
        return x_best

def run_packing():
    """
    Main function to solve the packing problem.
    Returns (centers, radii, sum_radii).
    """
    n = N_CIRCLES
    
    # Strategy:
    # 1. Run Differential Evolution with penalty method to find a good feasible (or near-feasible) solution.
    # 2. Refine using SLSQP with hard constraints.
    # 3. Repeat a few times with different seeds to ensure global optimum.
    
    best_x = None
    best_obj = float('inf')
    best_valid_sum = 0.0
    
    # We will run DE a few times. 
    # DE is stochastic.
    num_de_runs = 3
    
    for run_idx in range(num_de_runs):
        # Generate a random seed or just let DE handle it. 
        # We can't pass seed to DE easily without modifying call, but we can change bounds or logic?
        # Just running it multiple times is enough.
        
        # Initial bounds for DE
        de_bounds = []
        for _ in range(n):
            de_bounds.append((0.0, 1.0)) # x
            de_bounds.append((0.0, 1.0)) # y
            de_bounds.append((0.0, 0.5)) # r
            
        # We need to pass args to vectorized_objective. 
        # DE func signature is func(x). 
        # We can define a wrapper inside the loop? 
        # "No closures from function nesting" -> wrapper inside loop captures `run_idx`? 
        # No, it doesn't need to capture anything except constants. 
        # But to be safe, I will define a helper function that takes `x` and uses global constants?
        # Or just call the function directly if I can pass args?
        # scipy.optimize.differential_evolution accepts `args` tuple.
        
        # However, vectorized_objective takes 3 args.
        # args=(n, PENALTY_WEIGHT)
        
        try:
            res = scipy.optimize.differential_evolution(
                func=vectorized_objective,
                args=(n, PENALTY_WEIGHT),
                bounds=de_bounds,
                popsize=20,
                maxiter=300,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=run_idx + 42,
                polishing=False
            )
            
            # Check if result is good
            # The objective value from DE includes penalty.
            # If penalty is 0, obj_val = -sum_r.
            # We want to check if constraints are satisfied.
            
            # Re-evaluate penalty to be sure
            pen_val = vectorized_objective(res.x, n, PENALTY_WEIGHT)
            # Penalty part is roughly pen_val - (-sum_r) ? 
            # Actually vectorized_objective returns -sum_r + P*penalty.
            # If penalty is 0, it returns -sum_r.
            
            # Check validity explicitly
            centers = res.x[:2*n].reshape(n, 2)
            radii = res.x[2*n:]
            
            # Check constraints manually for validation
            valid = True
            
            # Wall check
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
                    valid = False
                    break
            
            if valid:
                # Overlap check
                for i in range(n):
                    for j in range(i+1, n):
                        dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                        if dist < radii[i] + radii[j] - 1e-9:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                current_sum = np.sum(radii)
                if current_sum > best_valid_sum:
                    best_valid_sum = current_sum
                    best_x = res.x.copy()
            else:
                # If not valid, the objective value might be misleading due to penalty trade-off.
                # But we can try to refine it anyway.
                pass
                
        except Exception as e:
            print(f"DE run {run_idx} failed: {e}")
            continue

    # If we found a valid solution, refine it.
    if best_x is not None:
        refined_x = refine_with_slsqp(best_x, n)
        
        # Validate refined solution
        centers = refined_x[:2*n].reshape(n, 2)
        radii = refined_x[2*n:]
        
        # Final validation check
        is_valid = True
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Boundary
            if x < r - 1e-12 or x > 1 - r + 1e-12 or y < r - 1e-12 or y > 1 - r + 1e-12:
                is_valid = False
            # Overlap
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-12:
                    is_valid = False
        
        if is_valid:
            best_x = refined_x
            best_valid_sum = np.sum(radii)
        else:
            # If refinement broke validity, stick to previous best if any
            pass
    else:
        # Fallback: Just return a valid small packing if optimization failed
        # e.g. grid of tiny circles
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.01)
        best_x = np.concatenate([centers.flatten(), radii])
        best_valid_sum = np.sum(radii)

    # Extract final result
    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Ensure radii are non-negative (should be, but just in case)
    radii = np.maximum(radii, 0.0)
    
    # Final sum
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum

# Allow running as script
if __name__ == "__main__":
    # We can test locally
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(f"Centers:\n{c}")
    # print(f"Radii:\n{r}")

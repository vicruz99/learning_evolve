# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ada29bac) state=b70fa7af sum of radii=1.975161 correctness=1.0
# stdout(first 200): warning: basinhopping: local minimization failure basinhopping step 0: f -1.97516 warning: basinhopping: local minimization failure basinhopping step 1: f -1.63797 trial_f -1.63797 accepted True lowes
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, basinhopping
import math

def solve_radii(centers):
    """
    Solves for the maximum sum of radii for a fixed set of centers using Linear Programming.
    """
    n = centers.shape[0]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c = np.ones(n) * -1.0
    
    # 1. Bounds for radii based on boundaries (x, 1-x, y, 1-y)
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        # Margin to the nearest wall
        margin = min(x, 1.0 - x, y, 1.0 - y)
        # Radius must be non-negative and within margin
        # If center is outside [0,1], margin is negative, which is handled by linprog
        # but we clamp to 0 to avoid infeasible bounds (0, neg)
        max_r = max(0.0, margin)
        bounds_r.append((0.0, max_r))
    
    # 2. Pairwise constraints: r_i + r_j <= dist(i, j)
    # Number of pairs
    m = n * (n - 1) // 2
    
    # Efficient construction of constraints
    # We can use a list of (i, j, dist) and convert to sparse or dense
    # Given n=26, dense matrix 325x26 is small enough.
    
    # Preallocate for speed
    # We will build rows for the constraint matrix A_ub
    # Since linprog takes A_ub (dense or sparse), let's build it.
    
    # Using a simple list comprehension for distance calculation might be faster than loops
    # But explicit loops are clear and fast enough for n=26
    
    # We can also skip constructing the full matrix if we use a specialized callback, 
    # but linprog needs it upfront.
    
    # Optimization: compute distance matrix first
    # dists[i, j]
    # Using broadcasting
    c_diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(c_diff ** 2, axis=2))
    
    # We only need upper triangle
    # Create A_ub and b_ub
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            # Constraint: 1*r_i + 1*r_j <= d
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    # Solve LP
    # method='highs' is efficient and default in recent scipy
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            # If infeasible (shouldn't happen with valid bounds), return 0
            return 0.0, np.zeros(n)
    except Exception:
        return 0.0, np.zeros(n)

def objective_function(centers_flat):
    """
    Objective function for the optimizer.
    Minimizes negative sum of radii.
    """
    centers = centers_flat.reshape(-1, 2)
    
    # Clamp centers to [0, 1] to keep them valid during optimization steps
    # This prevents LP from receiving invalid centers that might cause issues
    # although bounds in basinhopping should handle it, this is a safety measure.
    # Actually, basinhopping minimizer steps might go outside bounds if not careful.
    # We rely on bounds in the minimizer.
    
    sum_radii, radii = solve_radii(centers)
    
    # We want to maximize sum_radii, so minimize negative sum
    return -sum_radii, radii # return radii for tracking? No, objective function returns scalar.
    
    # Wait, basinhopping needs a function that returns scalar.
    # We can't return radii easily unless we wrap it.
    # Let's just return the value.

def minimizer_func(centers_flat):
    val, radii = objective_function(centers_flat)
    # Store best radii globally? 
    # Better to just return value.
    return val

def run_packing():
    n = 26
    
    # Initial configuration: Hexagonal grid
    # We want to place 26 points.
    # Let's try to generate a hex grid and pick 26 points.
    # Spacing s.
    # Rows: 5.
    # Row lengths: 6, 5, 6, 5, 4 (sum 26)
    
    centers = []
    row_height = 1.0 / (5.0) # rough estimate, we will optimize
    # Actually, let's just put them in a grid and let optimizer fix it.
    # A simple grid initialization is robust.
    
    # 5 rows, 6 cols -> 30 points. Take first 26.
    # Or better, a hex pattern.
    
    # Let's generate a hex grid of points
    # x = k * dx + (row_idx % 2) * dx/2
    # y = row_idx * dy
    
    # Let's estimate spacing to fit 26 circles.
    # Area per circle approx 1/26.
    # r approx 0.1. Diameter 0.2.
    # Spacing 0.2.
    
    dx = 0.15
    dy = dx * math.sqrt(3) / 2
    
    row_idx = 0
    while len(centers) < n:
        y = 0.1 + row_idx * dy
        if y > 0.9: # boundary check for generation
             # shift y to fit better?
             y = 0.9
        x_start = 0.1 if row_idx % 2 == 0 else 0.1 + dx/2
        
        k = 0
        while True:
            x = x_start + k * dx
            if x > 0.9:
                break
            if len(centers) < n:
                centers.append([x, y])
            k += 1
        row_idx += 1
        
        # Safety break
        if row_idx > 10:
            break
            
    # If not enough points (unlikely with this logic), pad with random
    while len(centers) < n:
        centers.append([np.random.rand(), np.random.rand()])
        
    centers = np.array(centers[:n])
    
    # Define bounds for the minimizer (centers must be in [0, 1])
    # 26 circles * 2 coords = 52 variables
    bounds_opt = [(0, 1)] * (n * 2)
    
    # Use BasinHopping for global optimization
    # Minimizer inside basin hopping
    minimizer_kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 100, "xatol": 1e-6, "fatol": 1e-6}}
    
    # Define a wrapper that only returns scalar for basinhopping
    def scalar_obj(centers_flat):
        return minimizer_func(centers_flat)
        
    try:
        # BasinHopping might be slow, limit niter
        # niter=50 might be enough if start is good
        ret = basinhopping(scalar_obj, centers.flatten(), minimizer_kwargs=minimizer_kwargs, 
                           niter=30, stepsize=0.05, interval=50, disp=True)
        
        best_centers = ret.x.reshape(-1, 2)
        # Retrieve best radii
        sum_rad, best_radii = solve_radii(best_centers)
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial
        sum_rad, best_radii = solve_radii(centers)
        best_centers = centers

    # Final validation and cleanup
    # Ensure radii are valid
    # The LP solver guarantees validity relative to centers.
    # We just return the result.
    
    # Note: best_centers might have tiny numerical errors outside [0,1] due to optimization?
    # But bounds were [0,1].
    
    return best_centers, best_radii, float(sum_rad)

if __name__ == "__main__":
    # Just to allow running as script for testing if needed, 
    # though the problem asks for run_packing function.
    pass

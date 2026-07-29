# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 93a6f440) state=c22b515d sum of radii=2.500974 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize, differential_evolution

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Objective: Minimize -r (to maximize r)
    # Variables: [x1, y1, ..., x26, y26, r]
    # Constraints: 
    #   0 <= r
    #   r <= x_i <= 1-r
    #   r <= y_i <= 1-r
    #   dist(i, j) >= 2r
    
    def objective(vars):
        # vars[:-1] are coords, vars[-1] is r
        r = vars[-1]
        return -r

    def boundary_constraints(vars):
        r = vars[-1]
        coords = vars[:-1].reshape(n, 2)
        # Constraints: r <= x <= 1-r => x - r >= 0, x - (1-r) <= 0 => 1-r-x >= 0
        # scipy constraints: fun(x) >= 0
        # So: x - r >= 0  => x - r
        #     (1-r) - x >= 0 => 1 - r - x
        
        cons = []
        for i in range(n):
            x, y = coords[i]
            # x - r >= 0
            cons.append(x - r)
            # y - r >= 0
            cons.append(y - r)
            # 1 - x - r >= 0
            cons.append(1 - x - r)
            # 1 - y - r >= 0
            cons.append(1 - y - r)
        return np.array(cons)

    def separation_constraints(vars):
        r = vars[-1]
        coords = vars[:-1].reshape(n, 2)
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((coords[i] - coords[j])**2)
                # dist >= 2r  => dist_sq >= 4r^2 => dist_sq - 4r^2 >= 0
                cons.append(dist_sq - 4 * r**2)
        return np.array(cons)

    # Initial guess:
    # Try a hexagonal-ish grid or just a grid with some noise.
    # 26 circles. 5x5 is 25. Let's try to fit 26 in a 6x5 grid pattern but optimized.
    # Or just a random perturbation of a grid.
    
    # Let's create a grid of 5 columns and 6 rows (30 points) and pick 26?
    # No, let's just use a dense packing guess.
    # A simple strategy: place points in a grid 5x6, scale to fit, then optimize.
    # Actually, let's use a random initialization near a grid to avoid bad local minima.
    
    np.random.seed(42)
    
    # Generate initial positions
    # Try to fit 26 points in [0,1]x[0,1] with some spacing
    # A 5x5 grid has spacing 0.25.
    # Let's try to arrange them in rows.
    # Maybe 6 rows?
    # Row counts: 5, 5, 5, 5, 5, 1?
    
    initial_centers = []
    # 5 rows of 5
    for r_idx in range(5):
        for c_idx in range(5):
            x = 0.1 + c_idx * 0.2
            y = 0.1 + r_idx * 0.2
            initial_centers.append([x, y])
    # 1 extra circle, place it in a gap?
    # Gap at (0.2, 0.2) maybe? But occupied.
    # Gap between (0.1, 0.1) and (0.3, 0.1)?
    # Let's place it at (0.2, 0.9) or something free?
    # Actually, random placement for the extra one might be safer or just add to a new row.
    # Let's add it at (0.95, 0.95)
    initial_centers.append([0.95, 0.95])
    
    initial_centers = np.array(initial_centers[:n])
    
    # Initial radius guess. 0.1 is a safe lower bound.
    r_guess = 0.09
    
    x0 = np.concatenate([initial_centers.flatten(), [r_guess]])
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': separation_constraints}
    ]
    
    # Bounds for x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)]
    
    # We might need to try a few restarts or a global optimizer.
    # SLSQP is local. Let's try running it a few times.
    
    best_result = None
    best_fun = -np.inf
    
    for _ in range(10): # 10 restarts
        # Perturb initial positions slightly
        noise = np.random.uniform(-0.05, 0.05, 2*n)
        current_x0 = x0.copy()
        current_x0[:2*n] = np.clip(current_x0[:2*n] + noise, 0.01, 0.99)
        
        try:
            res = minimize(objective, current_x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success and -res.fun > best_fun:
                best_fun = -res.fun
                best_result = res
        except:
            pass

    # If SLSQP fails or finds poor solution, fallback to a simple valid packing
    # E.g. equal circles in a grid.
    # But let's assume the optimizer works somewhat.
    
    if best_result is not None:
        final_vars = best_result.x
        final_centers = final_vars[:2*n].reshape(n, 2)
        final_r = final_vars[-1]
        
        # Validation check just in case
        # If constraints are violated slightly due to numerical issues, we might need to shrink r slightly
        # But the optimizer should respect them.
        
        # To be safe, check constraints and reduce r if needed
        # Check separation
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                if d < min_dist:
                    min_dist = d
        
        max_possible_r = min_dist / 2
        
        # Check boundary
        for i in range(n):
            x, y = final_centers[i]
            max_possible_r = min(max_possible_r, x, 1-x, y, 1-y)
            
        # Clamp r to feasible value
        final_r = min(final_r, max_possible_r * 0.999) # slightly smaller to be safe
        final_r = max(final_r, 0.0)
        
        final_radii = np.full(n, final_r)
        sum_radii = np.sum(final_radii)
        
        return final_centers, final_radii, sum_radii
    else:
        # Fallback: 5x5 grid plus one small circle?
        # Or just random small circles.
        # Let's try a grid.
        centers = []
        # 5x5 grid
        for r in range(5):
            for c in range(5):
                centers.append([0.1 + c*0.2, 0.1 + r*0.2])
        # 26th circle at a corner or edge?
        # 5x5 grid uses [0.1, 0.9].
        # We can squeeze in another one?
        # Maybe at (0.5, 0.5) but that's occupied.
        # Let's place it at (0.05, 0.05) with small radius?
        # Or just scale everything down.
        # 26 circles in 5x6 grid (scaled).
        # 6 rows, 5 cols?
        # Or just uniform grid.
        # 26 ~ 5.1^2.
        # Let's place in 6 rows.
        # Rows: 5, 5, 5, 5, 5, 1
        centers = []
        y_step = 1.0 / 7 # 7 gaps for 6 items? No.
        # Just uniform grid 6x5 (30 points), take first 26.
        # This is a valid packing for small r.
        # We want to maximize r.
        # If we use 6x5 grid, r = 1/(2*6) = 1/12 ~ 0.0833.
        # Sum = 26 * 0.0833 = 2.16.
        # This is valid but maybe not optimal.
        # However, the optimizer should have found better.
        # Let's return the grid packing as a safe fallback.
        
        centers = []
        # 5 columns, 6 rows
        # x: 0.1, 0.3, 0.5, 0.7, 0.9
        # y: 0.0833, 0.25, 0.4166, 0.5833, 0.75, 0.9166
        # Radius 0.0833
        r = 0.08
        for i in range(6):
            for j in range(5):
                if len(centers) < n:
                    cx = 0.1 + j * 0.2
                    cy = 0.1 + i * 0.15 # spacing 0.15?
                    # Better to just fill grid properly.
                    pass
        
        # Simple valid fallback:
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.01)
        return centers, radii, np.sum(radii)

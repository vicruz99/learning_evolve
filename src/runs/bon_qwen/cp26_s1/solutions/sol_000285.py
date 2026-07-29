# sol_000285 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5952a474) state=444eabac sum of radii=2.615196 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import time

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # Objective function: Minimize negative sum of radii (Maximize sum of radii)
    def objective(x):
        # x is flattened [x1, y1, r1, x2, y2, r2, ...]
        radii = x[2::3]
        return -np.sum(radii)

    # Constraints
    # 1. Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
    # 2. Distance constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    
    # We will construct the constraints list dynamically
    
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n

    def get_constraints():
        constraints = []
        
        # Boundary constraints for each circle
        for i in range(n):
            idx_x = 3 * i
            idx_y = 3 * i + 1
            idx_r = 3 * i + 2
            
            # x >= r  => x - r >= 0
            def con_x_min(x, i=i):
                return x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': con_x_min})
            
            # x <= 1 - r => 1 - x - r >= 0
            def con_x_max(x, i=i):
                return 1.0 - x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': con_x_max})
            
            # y >= r => y - r >= 0
            def con_y_min(x, i=i):
                return x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': con_y_min})
            
            # y <= 1 - r => 1 - y - r >= 0
            def con_y_max(x, i=i):
                return 1.0 - x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': con_y_max})
            
            # r >= 0 is handled by bounds, but strictly r >= epsilon might help?
            # Bounds handle r >= 0.

        # Distance constraints
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = 3 * i
                idx_j = 3 * j
                
                def con_dist(x, i=i, j=j):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    # Squared distance >= Squared sum of radii
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                
                constraints.append({'type': 'ineq', 'fun': con_dist})
        
        return constraints

    constraints = get_constraints()

    # Helper to generate initial guess
    def generate_initial_guess(seed=0):
        rng = np.random.RandomState(seed)
        
        # Try to place circles in a hexagonal-like grid first
        # We want to distribute 26 points in [0,1]x[0,1]
        
        # Let's create a grid of potential spots and pick 26
        # A 6x5 grid gives 30 spots.
        
        # Base grid points
        x_grid = np.linspace(0.1, 0.9, 6)
        y_grid = np.linspace(0.1, 0.9, 5)
        
        points = []
        for y in y_grid:
            for x in x_grid:
                points.append([x, y])
        
        # We have 30 points. We need 26.
        # Shuffle and pick 26 to break symmetry, or remove specific ones.
        # Removing corners might be good?
        # Let's just shuffle indices and pick 26.
        indices = list(range(30))
        rng.shuffle(indices)
        selected_indices = indices[:26]
        
        centers = np.array([points[k] for k in selected_indices])
        
        # Add small random perturbation to avoid symmetry locks
        centers += rng.uniform(-0.02, 0.02, size=centers.shape)
        
        # Clip to valid range slightly away from boundaries to start feasible
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initial radii: small value to ensure non-overlap initially
        # Max possible radius is approx 0.1. Start at 0.01.
        radii = np.full(n, 0.02)
        
        # Flatten vector
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]
            
        return x0

    best_x = None
    best_val = -np.inf
    
    # Run optimization multiple times with different seeds
    num_restarts = 5
    
    for k in range(num_restarts):
        x0 = generate_initial_guess(seed=k)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success or (res.fun < best_val): # Note: res.fun is negative sum
                # Check validity manually just in case
                # Although constraints should handle it, numerical issues might occur
                # But for the purpose of the function, we trust scipy if it terminates
                pass
            
            # The value we want to maximize is -res.fun
            current_sum_radii = -res.fun
            
            if current_sum_radii > best_val:
                best_val = current_sum_radii
                best_x = res.x.copy()
                
        except Exception as e:
            # If one run fails, continue with others
            pass

    if best_x is None:
        # Fallback: return a valid trivial solution if optimization failed completely
        # e.g. tiny circles
        best_x = generate_initial_guess(seed=42)
        best_val = 0.52 # 26 * 0.02

    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
    
    # Post-processing: ensure radii are non-negative (should be guaranteed by bounds)
    radii = np.maximum(radii, 0.0)
    
    # Final validation check (optional but good for debugging)
    # We won't print to stdout to keep it clean, but logic holds.
    
    return centers, radii, np.sum(radii)

# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 88e7083c) state=8420fbb3 sum of radii=2.090617 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Helper function to solve for optimal radii given centers using Linear Programming
    def solve_radii_lp(centers):
        # Objective: maximize sum(r_i) => minimize -sum(r_i)
        c_obj = -np.ones(n)
        
        A_ub = []
        b_ub = []
        
        # Precompute distances to avoid redundant calculations
        # r_i + r_j <= dist(i, j)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
                
        # Boundary constraints:
        # r_i <= x_i
        # r_i <= 1 - x_i
        # r_i <= y_i
        # r_i <= 1 - y_i
        for i in range(n):
            x, y = centers[i]
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(x) # r_i <= x
            
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(1 - x) # r_i <= 1 - x
            
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(y) # r_i <= y
            
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(1 - y) # r_i <= 1 - y
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, 0.5) for _ in range(n)] # Radii between 0 and 0.5
        
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x
        else:
            # Fallback if LP fails (should rarely happen with valid inputs)
            return np.full(n, 0.01)

    # Helper function for position optimization
    # Minimizes penalty for overlaps and boundary violations
    def position_penalty(params, radii):
        centers = params.reshape(n, 2)
        penalty = 0.0
        
        # Boundary penalties (soft constraints to keep circles inside)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # If center is too close to boundary relative to radius
            # Ideally x >= r and x <= 1-r.
            # Violation amount: max(0, r - x), max(0, r - (1-x))
            if x < r: penalty += (r - x)**2 * 1000
            if x > 1 - r: penalty += (x - (1 - r))**2 * 1000
            if y < r: penalty += (r - y)**2 * 1000
            if y > 1 - r: penalty += (y - (1 - r))**2 * 1000
            
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                r_sum = radii[i] + radii[j]
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < r_sum:
                    overlap = r_sum - dist
                    penalty += overlap**2 * 1000
        return penalty

    # 1. Initialization
    # Start with a hexagonal-like grid perturbation
    centers = np.zeros((n, 2))
    idx = 0
    # Try to fill rows
    # 5 rows of 5 or 6 circles
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    # Total 26.
    # Let's just place them evenly.
    
    # Simple grid initialization
    # 6 columns, 5 rows -> 30 spots. Pick 26.
    # Spacing 1/6 approx.
    pts = []
    for r in range(6):
        for c in range(5):
            if len(pts) >= 26: break
            x = (c + 0.5) / 5.0
            y = (r + 0.5) / 6.0
            pts.append([x, y])
        if len(pts) >= 26: break
    
    centers = np.array(pts[:n])
    # Add small random noise to break symmetry
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.01, 0.99) # Keep away from exact boundaries initially
    
    radii = np.full(n, 0.05) # Initial guess
    
    # 2. Iterative Optimization
    # Alternate between optimizing radii (LP) and positions (Nonlinear)
    for step in range(15): # Run multiple steps
        # Solve for optimal radii
        optimal_radii = solve_radii_lp(centers)
        
        # Optimize positions to relieve pressure
        # We pass optimal_radii to the penalty function
        x0 = centers.flatten()
        bounds_pos = [(0, 1)] * (2 * n)
        
        # Use L-BFGS-B
        res = opt.minimize(position_penalty, x0, args=(optimal_radii,), method='L-BFGS-B', 
                           bounds=bounds_pos, options={'maxiter': 200, 'ftol': 1e-10})
        
        centers = res.x.reshape(n, 2)
        # Ensure centers are valid
        centers = np.clip(centers, 0, 1)
        
        # Update radii for next step (they might have changed if we re-solved, 
        # but here we keep them from LP solution, positions moved to accommodate them better)
        # Actually, after moving positions, radii could potentially be larger.
        # The next iteration's LP will find them.
        
    # Final calculation
    final_radii = solve_radii_lp(centers)
    final_sum = np.sum(final_radii)
    
    return centers, final_radii, final_sum

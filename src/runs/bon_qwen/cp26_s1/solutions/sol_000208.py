# sol_000208 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c1389c4d) state=c496eb1e sum of radii=2.610743 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.ndarray of shape (26, 2)
        radii: np.ndarray of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # Initial configuration: Hexagonal packing
    # We aim for 5 rows. To fit 26 circles, we distribute them as 6, 5, 5, 5, 5.
    # This creates a dense, staggered lattice.
    row_counts = [6, 5, 5, 5, 5]
    
    centers = []
    for r_idx, count in enumerate(row_counts):
        # Calculate vertical position
        # Vertical spacing in hexagonal packing is r * sqrt(3)
        # We normalize spacing to fit within [0, 1] initially
        y = (r_idx + 0.5) * (1.0 / 5.0) # Uniform distribution for init
        
        # Calculate horizontal positions
        # Stagger every other row
        if r_idx % 2 == 0:
            # Row with 6 circles, starting closer to left
            # Width needs to accommodate 6 circles. 
            # Spacing approx 1/6. 
            offset = 0
        else:
            # Row with 5 circles, shifted
            offset = 1.0 / 12.0 # Half of spacing for 6 circles
            
        for c_idx in range(count):
            x = (c_idx + 0.5) * (1.0 / 6.0) + offset
            # Keep within bounds for initialization
            x = np.clip(x, 0.05, 0.95)
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Initial radii estimate
    # For N=26, sum ~ 2.636 implies average r ~ 0.101
    # We start with a safe radius that satisfies non-overlap for this layout
    initial_radii = np.full(n, 0.09)
    
    # Optimization function
    def objective(x):
        # x contains centers (n, 2) and radii (n,) flattened
        # Actually, let's keep centers and radii separate or flatten carefully.
        # Here we optimize a single vector of size 2*n + n = 78
        # But scipy.optimize.minimize works better with a single vector.
        # Let's restructure: x[:2*n] are centers, x[2*n:] are radii.
        pass

    # Better to define a solver that updates centers and radii
    # We will use a wrapper function
    
    def solve_packing(init_centers, init_radii):
        # Combine into a single optimization variable
        # x = [x1, y1, x2, y2, ..., r1, r2, ...]
        x0 = np.concatenate([init_centers.flatten(), init_radii])
        
        # Number of variables
        dim = len(x0)
        
        # Bounds:
        # x, y in [0, 1]
        # r in [0, 0.5] (theoretical max)
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # x_i
            bounds.append((0.0, 1.0)) # y_i
        for i in range(n):
            bounds.append((1e-6, 0.5)) # r_i
            
        constraints = []
        
        # Constraints: 
        # 1. Circle inside square: r <= x <= 1-r, r <= y <= 1-r
        #    => x - r >= 0
        #    => 1 - x - r >= 0
        #    => y - r >= 0
        #    => 1 - y - r >= 0
        
        def make_boundary_constraint(i):
            def constraint(val):
                x_val = val[2*i]
                y_val = val[2*i+1]
                r_val = val[2*n + i]
                # We need all 4 to be >= 0
                return np.array([
                    x_val - r_val,
                    1.0 - x_val - r_val,
                    y_val - r_val,
                    1.0 - y_val - r_val
                ])
            return constraint

        for i in range(n):
            # Add 4 constraints per circle
            # Note: scipy constraints can be a dict with 'fun' returning array
            # But 'SLSQP' handles vector constraints if we pass them carefully?
            # Actually, standard scipy minimize expects 'ineq' fun to return >= 0.
            # We can group them or add one by one. Adding one by one is safer for clarity
            # but slow. Let's add a vector constraint.
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i)})

        # 2. Non-overlap: dist(Ci, Cj) >= ri + rj
        #    => (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        
        def make_overlap_constraint(i, j):
            def constraint(val):
                xi, yi = val[2*i], val[2*i+1]
                xj, yj = val[2*j], val[2*j+1]
                ri, rj = val[2*n + i], val[2*n + j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                r_sum_sq = (ri + rj)**2
                return dist_sq - r_sum_sq
            return constraint

        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
                
        # Objective: Maximize sum of radii => Minimize negative sum
        def objective_func(val):
            radii = val[2*n:]
            return -np.sum(radii)
            
        # Run optimizer
        # 'SLSQP' is suitable for bound and inequality constrained problems
        res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
        
        if res.success:
            opt_centers = res.x[:2*n].reshape(n, 2)
            opt_radii = res.x[2*n:]
            return opt_centers, opt_radii
        else:
            # If fails, return best effort
            print("Optimization did not converge successfully.")
            opt_centers = res.x[:2*n].reshape(n, 2)
            opt_radii = res.x[2*n:]
            return opt_centers, opt_radii

    # Run the solver
    final_centers, final_radii = solve_packing(centers, initial_radii)
    
    # Post-processing: Ensure non-negative radii and clamp tiny negatives
    final_radii = np.maximum(final_radii, 0.0)
    
    # Validate and calculate sum
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

if __name__ == "__main__":
    import sys
    # Check if numpy is available, though environment should have it
    try:
        centers, radii, s_r = run_packing()
        print(f"Sum of radii: {s_r}")
        # Simple validation print
        print(f"Number of circles: {len(radii)}")
        # Check bounds
        for i in range(len(radii)):
            x, y = centers[i]
            r = radii[i]
            if x-r < -1e-9 or x+r > 1+1e-9 or y-r < -1e-9 or y+r > 1+1e-9:
                print(f"Warning: Circle {i} out of bounds")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

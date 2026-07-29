# sol_000167 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 403fd447) state=635c8302 sum of radii=2.614522 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constant for the number of circles
N_CIRCLES = 26

def objective_function(x):
    """
    Objective function to minimize.
    We want to maximize sum of radii, so we minimize negative sum of radii.
    x is a flattened array of length 3*N: [x1, y1, r1, x2, y2, r2, ...]
    """
    # Radii are at indices 2, 5, 8, ...
    radii = x[2::3]
    return -np.sum(radii)

def boundary_constraints(x):
    """
    Constraints ensuring circles are inside the unit square.
    x_i - r_i >= 0
    1 - x_i - r_i >= 0
    y_i - r_i >= 0
    1 - y_i - r_i >= 0
    Returns a vector of constraint values (must be >= 0).
    """
    # Unpack coordinates
    # x_coords at 0::3, y_coords at 1::3, radii at 2::3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Construct constraint vector
    # We need 4 constraints per circle
    cons = np.empty(4 * N_CIRCLES)
    
    # x >= r  => x - r >= 0
    cons[0:N_CIRCLES] = xs - rs
    
    # x <= 1-r => 1 - x - r >= 0
    cons[N_CIRCLES:2*N_CIRCLES] = 1.0 - xs - rs
    
    # y >= r => y - r >= 0
    cons[2*N_CIRCLES:3*N_CIRCLES] = ys - rs
    
    # y <= 1-r => 1 - y - r >= 0
    cons[3*N_CIRCLES:4*N_CIRCLES] = 1.0 - ys - rs
    
    return cons

def distance_constraints(x):
    """
    Constraints ensuring circles do not overlap.
    (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    Returns a vector of constraint values (must be >= 0).
    """
    # Unpack coordinates
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Generate all unique pairs (i, j) with i < j
    # Indices for xs, ys, rs
    indices = np.arange(N_CIRCLES)
    i_indices, j_indices = np.triu_indices(N_CIRCLES, k=1)
    
    # Coordinates for i and j
    xi = xs[i_indices]
    yi = ys[i_indices]
    ri = rs[i_indices]
    
    xj = xs[j_indices]
    yj = ys[j_indices]
    rj = rs[j_indices]
    
    # Calculate squared distance
    dx = xi - xj
    dy = yi - yj
    dist_sq = dx**2 + dy**2
    
    # Calculate squared sum of radii
    rad_sum = ri + rj
    rad_sum_sq = rad_sum**2
    
    # Constraint: dist_sq - rad_sum_sq >= 0
    return dist_sq - rad_sum_sq

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the optimization to pack 26 circles in a unit square.
    Returns (centers, radii, sum_radii).
    """
    
    # Define bounds for variables
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (radius cannot exceed 0.5 in unit square)
        
    # Define constraints for SLSQP
    # SLSQP requires constraints to be >= 0 for type 'ineq'
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': distance_constraints}
    ]
    
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # We will try multiple initializations to find a good local optimum
    n_attempts = 15
    
    for attempt in range(n_attempts):
        # --- Initialization Strategy ---
        
        if attempt < 8:
            # Strategy 1: Grid-based initialization (Hexagonal-like)
            # This tends to produce well-distributed starting points
            points = []
            # Try to create a dense grid
            # 6 rows, varying columns to approximate hexagonal packing
            rows = 6
            cols = 5
            
            # Generate grid points
            y_step = 1.0 / (rows + 1)
            x_step = 1.0 / (cols + 1)
            
            for r in range(rows):
                y = (r + 1) * y_step
                # Shift odd rows
                if r % 2 == 1:
                    x_offset = x_step / 2
                    # Might fit one less point if shifted significantly, but let's stick to logic
                    # Actually, let's just generate points and filter/select
                    for c in range(cols + 1):
                        x = x_offset + c * x_step
                        if 0 <= x <= 1:
                            points.append([x, y])
                else:
                    for c in range(cols + 1):
                        x = c * x_step
                        if 0 <= x <= 1:
                            points.append([x, y])
            
            # If we have too many or too few, adjust or shuffle
            # We need exactly 26
            if len(points) > N_CIRCLES:
                # Randomly select 26
                indices = np.random.choice(len(points), N_CIRCLES, replace=False)
                points = [points[i] for i in indices]
            elif len(points) < N_CIRCLES:
                # Fill remaining with random points
                while len(points) < N_CIRCLES:
                    points.append([np.random.rand(), np.random.rand()])
            
            centers = np.array(points)
            
        else:
            # Strategy 2: Pure random initialization
            centers = np.random.rand(N_CIRCLES, 2)
            
        # Initial radii: small but non-zero to allow optimization to start moving
        # If radii are 0, gradients for distance constraints might be weak initially
        initial_radii = np.full(N_CIRCLES, 0.02) 
        
        # Construct initial vector x0
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = initial_radii
        
        # Run Optimization
        try:
            res = minimize(
                objective_function, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or (res.nit > 0 and res.fun < 0): # res.fun is negative sum of radii
                current_sum = -res.fun
                
                # Extract solution
                cx = res.x[0::3]
                cy = res.x[1::3]
                cr = res.x[2::3]
                
                # Basic validation check (solver might violate slightly)
                # Check if radii are positive and centers valid
                if np.all(cr >= 0):
                    # Check boundaries
                    valid_boundaries = True
                    if np.any(cx < cr - 1e-9) or np.any(cx > 1 - cr + 1e-9): valid_boundaries = False
                    if np.any(cy < cr - 1e-9) or np.any(cy > 1 - cr + 1e-9): valid_boundaries = False
                    
                    if valid_boundaries:
                        # Check overlaps (strict check)
                        # We can do a quick check here
                        dists = np.sqrt((cx[:, None] - cx[None, :])**2 + (cy[:, None] - cy[None, :])**2)
                        rad_sums = cr[:, None] + cr[None, :]
                        
                        # Check lower triangle
                        # Use mask to ignore diagonal and upper triangle
                        mask = np.tril(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=-1)
                        
                        # dist >= rad_sum => dist - rad_sum >= 0
                        min_gap = np.min(dists[mask] - rad_sums[mask])
                        
                        # Allow tiny numerical error
                        if min_gap >= -1e-7:
                            if current_sum > best_sum_radii:
                                best_sum_radii = current_sum
                                best_centers = np.column_stack((cx, cy))
                                best_radii = cr
        except Exception:
            continue

    # Fallback if no valid solution found (unlikely)
    if best_centers is None:
        centers = np.random.rand(N_CIRCLES, 2)
        radii = np.full(N_CIRCLES, 0.01)
        # Ensure inside
        radii = np.minimum(radii, np.minimum(centers[:, 0], 1 - centers[:, 0]))
        radii = np.minimum(radii, np.minimum(centers[:, 1], 1 - centers[:, 1]))
        return centers, radii, np.sum(radii)
        
    return best_centers, best_radii, float(best_sum_radii)

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Quick validation print
    print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n_circles = 26
    np.random.seed(42)

    # Step 1: Find a good initial configuration using a hexagonal lattice
    # We search for the largest radius r such that we can fit at least 26 circles.
    best_r = 0.0
    best_centers = None
    
    # Search range for equal radius packing
    # Theoretical max for 26 circles is around 0.10-0.11
    for r_test in np.arange(0.11, 0.05, -0.001):
        # Generate hexagonal lattice points
        # Row spacing: r * sqrt(3)
        # Col spacing: 2 * r
        # Offset for odd rows: r
        
        points = []
        row = 0
        while True:
            y = r_test + row * r_test * np.sqrt(3)
            if y + r_test > 1.0:
                break
            
            # Determine x start
            if row % 2 == 0:
                x_start = r_test
            else:
                x_start = r_test + r_test # Shifted by r (since spacing is 2r, shift is r relative to grid? No. 
                                          # Standard hex packing: row 0 at x=r, 3r... row 1 at x=2r, 4r...
                                          # Actually shift is r. Center distance horizontally is 2r. 
                                          # Offset is r.
            
            col = 0
            while True:
                x = x_start + col * 2 * r_test
                if x + r_test > 1.0:
                    break
                
                # Check if point is inside [r, 1-r] for both coords
                if x - r_test >= -1e-9 and x + r_test <= 1.0 + 1e-9 and \
                   y - r_test >= -1e-9 and y + r_test <= 1.0 + 1e-9:
                    points.append((x, y))
                
                col += 1
            
            if len(points) >= n_circles:
                # We found enough points
                # Select top n_circles? They all have same r.
                # Just take the first n_circles
                selected = points[:n_circles]
                best_r = r_test
                best_centers = np.array(selected)
                break
            row += 1
        
        if best_centers is not None:
            break

    if best_centers is None:
        # Fallback to a simple grid if lattice search fails (unlikely)
        step = 1.0 / 6.0
        centers = []
        for i in range(6):
            for j in range(6):
                if len(centers) < n_circles:
                    centers.append([step + i * step, step + j * step])
        best_centers = np.array(centers)
        best_r = step / 2

    # Initialize radii
    radii = np.full(n_circles, best_r)

    # Step 2: Local Optimization
    # We want to maximize sum(radii). 
    # Variables: centers (26, 2) and radii (26,)
    # Total vars: 26 * 3 = 78.
    
    # Reshape to 1D array for optimizer
    x0 = np.concatenate([best_centers.flatten(), radii])
    
    def objective(vars_flat):
        centers = vars_flat[:52].reshape(26, 2)
        radii = vars_flat[52:]
        return -np.sum(radii) # Minimize negative sum

    def constraints_factory():
        constraints = []
        # Boundary constraints
        # x - r >= 0  => r - x <= 0
        # x + r <= 1  => x + r - 1 <= 0
        # y - r >= 0
        # y + r <= 1
        
        # We can implement these as simple bounds or constraints. 
        # scipy.optimize minimize supports bounds.
        pass
        return constraints

    # Bounds for variables
    # x in [0, 1]
    # y in [0, 1]
    # r in [0, 0.5] (max possible radius in square is 0.5)
    bounds = [(0, 1) for _ in range(52)] + [(1e-6, 0.5) for _ in range(26)]

    # Constraints function
    def constr_overlap(vars_flat):
        centers = vars_flat[:52].reshape(26, 2)
        radii = vars_flat[52:]
        vals = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # dist >= r_i + r_j  => r_i + r_j - dist <= 0
                vals.append(radii[i] + radii[j] - dist)
        return np.array(vals)

    def constr_boundary_x(vars_flat):
        centers = vars_flat[:52].reshape(26, 2)
        radii = vars_flat[52:]
        # x - r >= 0 => r - x <= 0
        # x + r <= 1 => x + r - 1 <= 0
        vals1 = radii - centers[:, 0]
        vals2 = radii + centers[:, 0] - 1.0
        return np.concatenate([vals1, vals2])

    def constr_boundary_y(vars_flat):
        centers = vars_flat[:52].reshape(26, 2)
        radii = vars_flat[52:]
        vals1 = radii - centers[:, 1]
        vals2 = radii + centers[:, 1] - 1.0
        return np.concatenate([vals1, vals2])

    cons = [
        {'type': 'ineq', 'fun': lambda v: -constr_overlap(v)}, # - (sum_r - dist) >= 0 => dist - sum_r >= 0
        {'type': 'ineq', 'fun': lambda v: -constr_boundary_x(v)},
        {'type': 'ineq', 'fun': lambda v: -constr_boundary_y(v)}
    ]

    # Use SLSQP solver
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            final_centers = res.x[:52].reshape(26, 2)
            final_radii = res.x[52:]
        else:
            # If optimization fails, return initial
            final_centers = best_centers
            final_radii = radii
    except Exception:
        final_centers = best_centers
        final_radii = radii

    # Clip radii to be non-negative
    final_radii = np.maximum(final_radii, 0)

    # Ensure centers are within [0,1]
    final_centers = np.clip(final_centers, 0, 1)

    # Adjust radii if circles are slightly outside due to optimization precision
    # This is a safety check
    for i in range(26):
        x, y = final_centers[i]
        r = final_radii[i]
        # If circle exceeds bounds, shrink it to fit
        if x - r < 0: r = x
        if x + r > 1: r = 1 - x
        if y - r < 0: r = y
        if y + r > 1: r = 1 - y
        final_radii[i] = max(r, 0)

    sum_radii = np.sum(final_radii)
    
    # Final validation check (optional but good for debugging)
    # We assume the optimizer did a good job, but let's ensure no NaNs
    if np.isnan(final_centers).any() or np.isnan(final_radii).any():
        # Fallback
        final_centers = best_centers
        final_radii = np.full(26, 0.09) # Safe fallback
        sum_radii = np.sum(final_radii)

    return final_centers, final_radii, float(sum_radii)
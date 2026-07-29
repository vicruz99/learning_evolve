# sol_000175 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ae68a5b3) state=b69b4647 sum of radii=2.491427 correctness=1.0
# stdout(first 200): Circles 2 and 13 overlap: dist=0.21444845329086443, r1+r2=0.21444845522984418
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    def compute_loss(params, n):
        """
        Objective function and constraint handler.
        Returns negative sum of radii (to be minimized) and constraint values.
        """
        # Unpack parameters: [x1, y1, r1, x2, y2, r2, ...]
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        # Objective: Maximize sum of radii
        obj = -np.sum(radii)
        
        # Constraint violations (penalty approach combined with SLSQP constraints)
        # We return an array of constraint values that must be >= 0
        constraints = []
        
        # 1. Boundary constraints: r <= x <= 1-r  =>  x - r >= 0, 1 - x - r >= 0
        # 2. Non-overlap: dist(i, j) >= r_i + r_j => dist^2 - (r_i + r_j)^2 >= 0
        
        # We'll accumulate these into a list for the 'ineq' constraint type
        # Note: scipy expects 'ineq' as g(x) >= 0
        
        # Boundary constraints (4 per circle)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            constraints.append(x - r)          # Left wall
            constraints.append(1.0 - x - r)    # Right wall
            constraints.append(y - r)          # Bottom wall
            constraints.append(1.0 - y - r)    # Top wall
            
            # Optional: Non-negative radius (usually handled by bounds, but for safety)
            constraints.append(r)
            
        # Overlap constraints (N*(N-1)/2)
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((centers[i] - centers[j])**2)
                radii_sum = radii[i] + radii[j]
                # We check dist^2 - (r_i + r_j)^2 >= 0
                constraints.append(dist_sq - radii_sum**2)
                
        return obj, np.array(constraints)

    def get_constraints_for_scipy(params, n):
        # Helper to extract just the constraint array for scipy
        _, cons = compute_loss(params, n)
        return cons

    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # We will try a few different initial configurations
    # 1. Hexagonal-like grid
    # 2. Random perturbation
    
    # Initialize centers for hexagonal pattern
    # Rows: 5, 4, 5, 4, 5, 3 (Total 26)
    # Spacing approx 0.15 vertically, 0.17 horizontally
    
    rows_config = [5, 4, 5, 4, 5, 3]
    y_coords = []
    current_y = 0.15
    for count in rows_config:
        for _ in range(count):
            y_coords.append(current_y)
        current_y += 0.15
        
    centers_init = []
    idx = 0
    y_val = 0.15
    row_idx = 0
    
    # Generate centers
    temp_centers = []
    y_pos = 0.12
    for count in rows_config:
        # x positions centered in [0, 1]
        if count > 0:
            # Spacing
            span = 1.0 - 0.2 # Leave some margin for radius
            if count > 1:
                step = span / (count - 1)
                x_start = 0.1 + (1.0 - 0.2 - (count-1)*step)/2
                xs = [x_start + k*step for k in range(count)]
            else:
                xs = [0.5]
            
            for x in xs:
                temp_centers.append([x, y_pos])
        
        # Shift y for next row
        y_pos += 0.14
        if row_idx % 2 == 1:
            # Shift x slightly for hexagonal effect? 
            # Actually, let's keep it simple grid first, optimizer will shift
            pass
        row_idx += 1
        
    # If we need exactly 26, check length
    while len(temp_centers) < 26:
        temp_centers.append([0.5, 0.5]) # Fallback
    centers_init = np.array(temp_centers[:26])
    
    # Initial radii
    radii_init = np.ones(n) * 0.08
    
    # Parameter vector
    params_init = np.concatenate([centers_init.flatten(), radii_init])
    
    # Bounds for parameters
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-6, 0.5)) # r (small positive lower bound)
        
    # Constraint definition for SLSQP
    # 'ineq' means g(x) >= 0
    cons = {'type': 'ineq', 'fun': lambda p: get_constraints_for_scipy(p, n)}
    
    # Optimization loop with restarts
    for restart in range(5):
        # Perturb initial guess slightly for different restarts
        if restart > 0:
            current_centers = centers_init.copy()
            # Add noise
            noise = np.random.randn(*current_centers.shape) * 0.05
            current_centers += noise
            # Clip
            current_centers = np.clip(current_centers, 0.05, 0.95)
            current_radii = radii_init + np.random.randn(n) * 0.02
            current_radii = np.clip(current_radii, 0.01, 0.4)
            params_start = np.concatenate([current_centers.flatten(), current_radii])
        else:
            params_start = params_init
            
        try:
            res = minimize(
                lambda p: compute_loss(p, n)[0],
                params_start,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'ftol': 1e-8, 'maxiter': 500, 'disp': False}
            )
            
            if res.success:
                final_params = res.x
                final_centers = final_params[:2*n].reshape(n, 2)
                final_radii = final_params[2*n:]
                current_sum = np.sum(final_radii)
                
                # Verify validity strictly
                if validate_packing(final_centers, final_radii):
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = final_centers.copy()
                        best_radii = final_radii.copy()
        except Exception:
            continue

    # If optimization failed to find a valid packing, fallback to a simple grid
    if best_sum < 0:
        # Fallback: 5x5 grid with smaller radius to fit 26?
        # Or just a dense grid
        centers_fb = np.zeros((26, 2))
        radii_fb = np.zeros(26)
        
        # Try to fit 26 circles in a 6x5 grid arrangement roughly
        # 5 rows of 5, 1 row of 1?
        # Let's do 5 rows of 5 (25) + 1 in center?
        # Actually, just place them on a grid and reduce radius until valid
        
        # Grid spacing
        grid_size = 6 # 6x6 grid has 36 spots
        step = 1.0 / (grid_size + 1)
        r_est = step * 0.4
        
        idx = 0
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                if idx < 26:
                    centers_fb[idx] = [i * step, j * step]
                    radii_fb[idx] = r_est
                    idx += 1
        
        # Adjust radius to be valid
        # Check overlaps
        # This is a rough fallback
        while True:
            valid = True
            for i in range(26):
                for j in range(i+1, 26):
                    dist = np.linalg.norm(centers_fb[i] - centers_fb[j])
                    if dist < radii_fb[i] + radii_fb[j]:
                        valid = False
                        # Reduce radii
                        radii_fb[i] *= 0.95
                        radii_fb[j] *= 0.95
            
            if valid:
                # Check boundaries
                for i in range(26):
                    x, y = centers_fb[i]
                    r = radii_fb[i]
                    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                        valid = False
                        radii_fb[i] *= 0.95
            
            if valid:
                break
            
            # Safety break
            if np.sum(radii_fb) < 1e-6:
                break
        
        best_centers = centers_fb
        best_radii = radii_fb
        best_sum = np.sum(radii_fb)

    return best_centers, best_radii, best_sum

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

# To run
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")

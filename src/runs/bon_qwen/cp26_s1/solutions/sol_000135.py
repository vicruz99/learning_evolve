# sol_000135 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0b92a944) state=2098d4dd sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    
    # Function to check validity (similar to validate_packing but used internally if needed)
    def is_valid(centers, radii):
        # Check bounds
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2
                r_sum = radii[i] + radii[j]
                if dist_sq < (r_sum - 1e-9)**2:
                    return False
        return True

    # Objective function: negative sum of radii (since minimize)
    def objective(vars):
        # vars shape: (3*n,) -> x1, y1, r1, x2, y2, r2, ...
        r_sum = 0.0
        for i in range(n):
            r_sum += vars[3*i + 2]
        return -r_sum

    # Constraints
    def get_constraints(n):
        constraints = []
        
        # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        # x - r >= 0  => vars[3i] - vars[3i+2] >= 0
        # 1 - x - r >= 0 => 1 - vars[3i] - vars[3i+2] >= 0
        # ...
        
        for i in range(n):
            # x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i] - v[3*i+2]
            })
            # 1 - x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]
            })
            # y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
            })
            # 1 - y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]
            })
            
            # r >= 0 (technically covered by boundary if x,y in [0,1], but good to be safe)
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+2]
            })

        # Overlap constraints: dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: \
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                })
        return constraints

    constraints_list = get_constraints(n)

    # Helper to create variables array from centers and radii
    def to_vars(centers, radii):
        vars = np.zeros(3 * n)
        for i in range(n):
            vars[3*i] = centers[i, 0]
            vars[3*i+1] = centers[i, 1]
            vars[3*i+2] = radii[i]
        return vars

    # Helper to extract centers and radii from variables
    def from_vars(vars):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = vars[3*i]
            centers[i, 1] = vars[3*i+1]
            radii[i] = vars[3*i+2]
        return centers, radii

    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    # Strategy 1: Hexagonal Grid Initialization
    def generate_hex_init(r_init=0.09):
        centers = np.zeros((n, 2))
        radii = np.full(n, r_init)
        idx = 0
        
        # Try to fit in a hexagonal pattern
        # Rows
        # Vertical spacing sqrt(3)*r
        # Horizontal spacing 2r, shift r for odd rows
        y = r_init
        row_idx = 0
        while idx < n and y + r_init <= 1.0:
            x_start = r_init + (row_idx % 2) * r_init
            x = x_start
            while x + r_init <= 1.0 and idx < n:
                centers[idx] = [x, y]
                idx += 1
                x += 2 * r_init
            y += np.sqrt(3) * r_init
            row_idx += 1
            
        # Fill remaining if any
        while idx < n:
            # Place in random valid spots or grid spots not taken
            # Simple grid fallback
            found = False
            for gy in np.arange(0.1, 1.0, 0.15):
                for gx in np.arange(0.1, 1.0, 0.15):
                    # Check if close to existing
                    close = False
                    for k in range(idx):
                        if np.hypot(centers[k,0]-gx, centers[k,1]-gy) < 0.2:
                            close = True
                            break
                    if not close:
                        centers[idx] = [gx, gy]
                        found = True
                        break
                if found: break
            if not found:
                # Just place somewhere safe, e.g. corner
                centers[idx] = [0.1, 0.1] # Will be fixed by optimizer
                radii[idx] = 0.05
            idx += 1
        return centers, radii

    # Strategy 2: Random Initialization with small radii
    def generate_random_init():
        centers = np.random.rand(n, 2)
        # Radii small to start
        radii = np.full(n, 0.05)
        return centers, radii

    # Run optimization multiple times with different seeds/initializations
    attempts = [
        generate_hex_init(0.09),
        generate_hex_init(0.10),
        generate_hex_init(0.08),
        generate_random_init(),
        generate_random_init(),
    ]

    for init_centers, init_radii in attempts:
        # Perturb slightly to avoid symmetry issues if exact grid
        if not np.allclose(init_centers, init_centers * 1.0001): 
             pass # Just a placeholder logic, actually we just use the init
        
        vars0 = to_vars(init_centers, init_radii)
        
        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r

        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds, 
                           constraints=constraints_list, options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                curr_centers, curr_radii = from_vars(res.x)
                # Clean up numerical noise
                curr_radii = np.maximum(curr_radii, 0.0)
                # Re-center if slightly out (clamp)
                for i in range(n):
                    r = curr_radii[i]
                    cx, cy = curr_centers[i]
                    # Clamp center to [r, 1-r]
                    cx = np.clip(cx, r, 1-r)
                    cy = np.clip(cy, r, 1-r)
                    curr_centers[i] = [cx, cy]
                
                # Verify validity (with slight tolerance)
                # The optimizer constraints might be satisfied within tolerance
                # We do a final check
                valid = True
                # Check bounds
                for i in range(n):
                    x, y = curr_centers[i]
                    r = curr_radii[i]
                    if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
                        valid = False
                        break
                    # Ensure radii non-negative
                    if r < 0: valid = False; break
                
                if valid:
                    # Check overlaps
                    for i in range(n):
                        for j in range(i+1, n):
                            d = np.hypot(curr_centers[i,0]-curr_centers[j,0], curr_centers[i,1]-curr_centers[j,1])
                            if d < curr_radii[i] + curr_radii[j] - 1e-7:
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        s = np.sum(curr_radii)
                        if s > best_sum_radii:
                            best_sum_radii = s
                            best_centers = curr_centers.copy()
                            best_radii = curr_radii.copy()
        except Exception as e:
            continue

    # If optimization failed to find valid, fallback to a simple valid packing
    if best_centers is None:
        # Simple grid fallback
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        # 5x5 grid r=0.1 gives 25 circles. 
        # We need 26.
        # Let's use r=0.08 for 6x5 grid (30 spots), pick 26.
        r_fallback = 0.08
        idx = 0
        for y in np.arange(0.1, 1.0, 0.16): # spacing approx 2r
            for x in np.arange(0.1, 1.0, 0.16):
                if idx < n:
                    best_centers[idx] = [x, y]
                    best_radii[idx] = r_fallback
                    idx += 1
                else:
                    break
            if idx >= n: break
        # Fill rest if needed
        while idx < n:
            best_centers[idx] = [0.1, 0.1]
            best_radii[idx] = 0.01
            idx += 1
        
        best_sum_radii = np.sum(best_radii)

    # Final adjustment: try to equalize radii slightly if they are very different?
    # Not strictly necessary if optimizer did its job.
    # But for stability, we can run a quick local move to increase radii uniformly if possible.
    # However, the constraints enforce max possible radii for the configuration.
    
    return best_centers, best_radii, best_sum_radii

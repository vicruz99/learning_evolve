# sol_000033 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b079e3ed) state=6c7790c5 sum of radii=2.511960 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # ---------------------------------------------------------
    # 1. Initialization Strategies
    # ---------------------------------------------------------
    def generate_hex_grid():
        """Generates a hexagonal grid layout."""
        centers = []
        # Try to fit 5 rows
        y_spacing = 0.1732 # approx sqrt(3)/10
        x_spacing = 0.2
        y = 0.1
        for row in range(6):
            x = 0.1
            # Shift every other row
            if row % 2 == 1:
                x += 0.1
            while x + 0.1 < 1.0 and len(centers) < n:
                centers.append([x, y])
                x += x_spacing
            y += y_spacing
        return np.array(centers[:n])

    def generate_greedy():
        """Greedy placement: place circle at largest valid radius."""
        centers = []
        radii = []
        # Grid search resolution
        res = 0.01
        xs = np.arange(0, 1.001, res)
        ys = np.arange(0, 1.001, res)
        
        for _ in range(n):
            best_r = 0
            best_pos = None
            
            # Simplified greedy: try grid points
            for x in xs:
                for y in ys:
                    r_min = min(x, 1-x, y, 1-y)
                    if r_min <= best_r: continue
                    
                    # Check against existing circles
                    valid = True
                    for cx, cy, cr in zip(centers, radii):
                        dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                        if dist < cr + r_min:
                            valid = False
                            break
                    
                    if valid and r_min > best_r:
                        best_r = r_min
                        best_pos = (x, y)
            
            if best_pos:
                centers.append(best_pos)
                radii.append(best_r)
            else:
                # Fallback if grid too coarse
                centers.append([np.random.rand(), np.random.rand()])
                radii.append(0.01)
                
        return np.array(centers), np.array(radii)

    # ---------------------------------------------------------
    # 2. Optimization Function
    # ---------------------------------------------------------
    def optimize_from(initial_centers, initial_radii):
        """Optimizes a configuration using L-BFGS-B with penalty method."""
        
        # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i]   = initial_centers[i, 0]
            x0[3*i+1] = initial_centers[i, 1]
            x0[3*i+2] = initial_radii[i]
            
        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
        # Penalty weight
        W = 5000.0
        
        def objective(vars):
            pts = vars.reshape(n, 3)
            x = pts[:, 0]
            y = pts[:, 1]
            r = pts[:, 2]
            
            # Objective: Maximize sum of radii
            loss = -np.sum(r)
            
            # 1. Wall Penalties
            # x >= r, x <= 1-r, y >= r, y <= 1-r
            # Violations: r - x, x + r - 1, r - y, y + r - 1
            wall_viol = np.maximum(0, r - x) + np.maximum(0, x + r - 1) + \
                        np.maximum(0, r - y) + np.maximum(0, y + r - 1)
            penalty = W * np.sum(wall_viol**2)
            
            # 2. Overlap Penalties
            # dist(i,j) >= r_i + r_j  =>  r_i + r_j - dist <= 0
            # Vectorized distance matrix
            # Using broadcasting for (N, 1) - (1, N)
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Radii sum matrix
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            
            # Overlap matrix
            overlaps = r_sum - dists
            
            # We only care about i < j to avoid double counting and self
            # Use upper triangular part
            tri_indices = np.triu_indices(n, k=1)
            overlap_viol = np.maximum(0, overlaps[tri_indices])
            penalty += W * np.sum(overlap_viol**2)
            
            return loss + penalty

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            best_vars = res.x
            best_centers = best_vars.reshape(n, 3)[:, :2]
            best_radii = best_vars.reshape(n, 3)[:, 2]
            
            # Clip negative radii just in case
            best_radii = np.maximum(best_radii, 0.0)
            
            return best_centers, best_radii
        except Exception:
            return initial_centers, initial_radii

    # ---------------------------------------------------------
    # 3. Run Multiple Starts and Select Best
    # ---------------------------------------------------------
    best_total_r = -1.0
    best_solution = (np.zeros((n, 2)), np.zeros(n))
    
    # Strategy 1: Hex Grid
    centers_hex = generate_hex_grid()
    radii_hex = np.full(n, 0.05) # Small initial radius to be expanded
    c, r = optimize_from(centers_hex, radii_hex)
    if np.sum(r) > best_total_r:
        best_total_r = np.sum(r)
        best_solution = (c, r)

    # Strategy 2: Greedy
    try:
        centers_greedy, radii_greedy = generate_greedy()
        c, r = optimize_from(centers_greedy, radii_greedy)
        if np.sum(r) > best_total_r:
            best_total_r = np.sum(r)
            best_solution = (c, r)
    except Exception:
        pass

    # Strategy 3: Random restarts
    np.random.seed(42)
    for _ in range(10):
        centers_rand = np.random.rand(n, 2)
        radii_rand = np.full(n, 0.01)
        c, r = optimize_from(centers_rand, radii_rand)
        if np.sum(r) > best_total_r:
            best_total_r = np.sum(r)
            best_solution = (c, r)

    final_centers, final_radii = best_solution
    
    # Final Safety Check & Cleanup
    # Ensure boundaries are respected strictly for the validation function
    # The optimizer might have tiny violations due to floating point, 
    # so we shrink radii slightly if needed.
    
    # Check bounds
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        # Max valid radius at this position
        r_max = min(x, 1-x, y, 1-y)
        if r > r_max + 1e-9:
            final_radii[i] = r_max
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
            r_sum = final_radii[i] + final_radii[j]
            if dist < r_sum - 1e-12:
                # Shrink radii to fit exactly
                scale = dist / r_sum
                final_radii[i] *= scale
                final_radii[j] *= scale

    return final_centers, final_radii, np.sum(final_radii)

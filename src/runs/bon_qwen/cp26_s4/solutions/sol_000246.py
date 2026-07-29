# sol_000246 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e7c70ed6) state=32f03353 sum of radii=2.540000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_sum = -np.inf
    best_centers = None
    best_radii = None

    def objective(x):
        # x contains [x1, y1, r1, x2, y2, r2, ...]
        radii = x[2::3]
        return -np.sum(radii)

    def boundary_constraints(x):
        cons = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            # Circle inside [0,1]x[0,1]
            # x - r >= 0  => x - r - eps >= 0
            # x + r <= 1  => 1 - x - r - eps >= 0
            cons.append(xi - ri - 1e-6)
            cons.append(1.0 - xi - ri - 1e-6)
            cons.append(yi - ri - 1e-6)
            cons.append(1.0 - yi - ri - 1e-6)
        return cons

    def non_overlap_constraints(x):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                # dist >= sum_r  => dist^2 >= sum_r^2
                cons.append(dist_sq - sum_r**2 - 1e-10)
        return cons

    def get_constraints(x):
        return boundary_constraints(x) + non_overlap_constraints(x)

    def generate_grid_init():
        # 5x5 grid plus one
        centers = []
        radii = []
        # 5x5 grid
        for r in range(5):
            for c in range(5):
                x = 0.1 + c * 0.2
                y = 0.1 + r * 0.2
                centers.append([x, y])
                radii.append(0.1)
        
        # Add one circle in a gap (center of the square hole at 0.2, 0.2)
        # But wait, 0.1 is radius of grid circles.
        # Gap at (0.2, 0.2) is distance 0.1414 from (0.1, 0.1)
        # Max radius there is approx 0.0414
        centers.append([0.2, 0.2])
        radii.append(0.04)
        
        # Flatten
        init_x = np.zeros(3 * n)
        for i in range(n):
            init_x[3*i] = centers[i][0]
            init_x[3*i+1] = centers[i][1]
            init_x[3*i+2] = radii[i]
        return init_x

    def generate_hex_init():
        centers = []
        radii = []
        r = 0.09
        # Hexagonal packing rows
        # Row 0: 5 circles
        for i in range(5):
            centers.append([r + i*2*r, r])
            radii.append(r)
        # Row 1: 5 circles (offset)
        for i in range(5):
            centers.append([2*r + i*2*r, r + r*np.sqrt(3)])
            radii.append(r)
        # Row 2: 5 circles
        for i in range(5):
            centers.append([r + i*2*r, 2*r + r*np.sqrt(3)])
            radii.append(r)
        # Row 3: 5 circles
        for i in range(5):
            centers.append([2*r + i*2*r, 3*r + r*np.sqrt(3)])
            radii.append(r)
        # Row 4: 5 circles
        for i in range(5):
            centers.append([r + i*2*r, 4*r + r*np.sqrt(3)])
            radii.append(r)
        # Row 5: 1 circle
        centers.append([0.5, 5*r + r*np.sqrt(3) + r]) # Adjust y to fit
        radii.append(r)
        
        # Resize to fit if necessary (simple heuristic)
        # Just take first 26
        centers = centers[:n]
        radii = radii[:n]
        
        # Center them in the square
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        # Scale to fit in [0.1, 0.9] roughly
        scale_x = 0.8 / width if width > 0 else 1
        scale_y = 0.8 / height if height > 0 else 1
        scale = min(scale_x, scale_y)
        
        # Apply scale and shift
        new_centers = []
        for c in centers:
            nc = [(c[0]-min_x)*scale + 0.1, (c[1]-min_y)*scale + 0.1]
            new_centers.append(nc)
        
        init_x = np.zeros(3 * n)
        for i in range(n):
            init_x[3*i] = new_centers[i][0]
            init_x[3*i+1] = new_centers[i][1]
            init_x[3*i+2] = radii[i] * scale
        return init_x

    def generate_random_init():
        np.random.seed(42)
        centers = np.random.rand(n, 2)
        radii = np.random.rand(n) * 0.05 + 0.05
        init_x = np.zeros(3 * n)
        for i in range(n):
            init_x[3*i] = centers[i, 0]
            init_x[3*i+1] = centers[i, 1]
            init_x[3*i+2] = radii[i]
        return init_x

    # List of initial guesses
    inits = [generate_grid_init(), generate_hex_init(), generate_random_init()]
    
    # Add some perturbed versions
    for _ in range(3):
        base = generate_grid_init()
        noise = np.random.normal(0, 0.01, size=base.shape)
        # Keep radii positive
        noise[2::3] = np.abs(noise[2::3]) * 0.05 
        inits.append(base + noise)

    bounds = [(0, 1), (0, 1), (0, 1)] * n # x, y in [0,1], r >= 0. r upper bound 1 is loose

    for i, x0 in enumerate(inits):
        try:
            res = opt.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': get_constraints},
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or res.fun < -best_sum:
                curr_sum = -res.fun
                # Validate constraints manually to be safe
                centers_opt = res.x[0::3].reshape(n, 2)
                radii_opt = res.x[2::3]
                
                # Quick check for validity (approx)
                valid = True
                # Check boundaries
                for k in range(n):
                    if radii_opt[k] < 1e-7: valid = False; break
                    if centers_opt[k, 0] - radii_opt[k] < -1e-5 or centers_opt[k, 0] + radii_opt[k] > 1 + 1e-5: valid = False; break
                    if centers_opt[k, 1] - radii_opt[k] < -1e-5 or centers_opt[k, 1] + radii_opt[k] > 1 + 1e-5: valid = False; break
                
                if valid:
                     # Check overlaps
                    for k in range(n):
                        for m in range(k+1, n):
                            d = np.sqrt(np.sum((centers_opt[k] - centers_opt[m])**2))
                            if d < radii_opt[k] + radii_opt[m] - 1e-5:
                                valid = False
                                break
                        if not valid: break
                
                if valid and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = centers_opt.copy()
                    best_radii = radii_opt.copy()
        except Exception as e:
            pass

    # Fallback to a valid simple solution if optimization fails
    if best_centers is None or np.isnan(best_centers).any():
        # 5x5 grid + 1 small
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        idx = 0
        for r in range(5):
            for c in range(5):
                best_centers[idx, 0] = 0.1 + c * 0.2
                best_centers[idx, 1] = 0.1 + r * 0.2
                best_radii[idx] = 0.1
                idx += 1
        best_centers[idx, 0] = 0.2
        best_centers[idx, 1] = 0.2
        best_radii[idx] = 0.04
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)

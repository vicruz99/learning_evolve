# sol_000309 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5f579b4b) state=13fe8369 sum of radii=2.620356 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # Function to calculate constraints and objective
    def objective_and_constraints(vars, n):
        # vars is a 1D array of size 3*n: [x0, y0, r0, x1, y1, r1, ...]
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        for i in range(n):
            centers[i, 0] = vars[3*i]
            centers[i, 1] = vars[3*i + 1]
            radii[i] = vars[3*i + 2]
            
        # Objective: maximize sum of radii -> minimize negative sum
        obj_val = -np.sum(radii)
        
        # Constraints
        constr = []
        
        # Non-overlap constraints: dist(i,j) >= ri + rj
        # Equivalent to: ri + rj - dist(i,j) <= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                constr.append(radii[i] + radii[j] - dist)
        
        # Boundary constraints
        for i in range(n):
            # x - r >= 0  => r - x <= 0
            constr.append(radii[i] - centers[i, 0])
            # x + r <= 1  => x + r - 1 <= 0
            constr.append(centers[i, 0] + radii[i] - 1.0)
            # y - r >= 0  => r - y <= 0
            constr.append(radii[i] - centers[i, 1])
            # y + r <= 1  => y + r - 1 <= 0
            constr.append(centers[i, 1] + radii[i] - 1.0)
            
        # Non-negative radii handled by bounds, but let's be safe or rely on bounds
        # SLSQP handles bounds well.
        
        return obj_val, constr

    def get_constraints_dict(n):
        # We need to return constraints in a format for scipy
        # Since we have many constraints, passing them as a list of dicts is better
        # But scipy.optimize.minimize with SLSQP accepts a list of dicts or a single dict with 'ineq'
        # However, passing a large number of scalar constraints might be slow.
        # Let's define a function that returns the values.
        
        def con_func(vars):
            centers = np.zeros((n, 2))
            radii = np.zeros(n)
            for i in range(n):
                centers[i, 0] = vars[3*i]
                centers[i, 1] = vars[3*i + 1]
                radii[i] = vars[3*i + 2]
            
            vals = []
            # Overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    vals.append(dist - radii[i] - radii[j]) # >= 0
            
            # Boundaries
            for i in range(n):
                vals.append(centers[i, 0] - radii[i])      # x >= r
                vals.append(1.0 - centers[i, 0] - radii[i]) # 1-x >= r
                vals.append(centers[i, 1] - radii[i])      # y >= r
                vals.append(1.0 - centers[i, 1] - radii[i]) # 1-y >= r
            
            return np.array(vals)

        return {'type': 'ineq', 'fun': con_func}

    # Initialization helpers
    def init_grid_perturbed(n):
        # 5x5 grid has 25, we need 26.
        # Let's try a 6x5 grid scaled down? Or 5x5 + 1.
        # 5x5 grid centers at 0.1, 0.3, 0.5, 0.7, 0.9. Radius 0.1.
        # Add 26th circle in a gap?
        centers = []
        radii = []
        
        # Base 5x5
        coords = [0.1, 0.3, 0.5, 0.7, 0.9]
        count = 0
        for x in coords:
            for y in coords:
                if count < n:
                    centers.append([x, y])
                    radii.append(0.1)
                    count += 1
        
        # If we need more, add in gaps
        while count < n:
            # Add at center of square if not present?
            # Or random perturbation
            cx, cy = 0.5, 0.5
            # Check if close to existing
            too_close = False
            for c in centers:
                if math.hypot(c[0]-cx, c[1]-cy) < 0.2:
                    too_close = True
                    break
            if not too_close:
                centers.append([cx, cy])
                radii.append(0.05) # Small radius
                count += 1
            else:
                # Just add a random point
                centers.append([np.random.rand(), np.random.rand()])
                radii.append(0.01)
                count += 1
                
        return centers, radii

    def init_hexagonal(n):
        centers = []
        radii = []
        r_est = 0.09 # Start small
        
        y = r_est
        row = 0
        count = 0
        
        while count < n:
            x = r_est
            shift = 0
            if row % 2 == 1:
                shift = r_est # Shift by radius for hex packing
            
            while x <= 1.0 - r_est:
                centers.append([x + shift, y])
                radii.append(r_est)
                count += 1
                if count >= n:
                    break
                x += 2 * r_est
            
            y += r_est * math.sqrt(3)
            row += 1
            # If y exceeds, we might stop, but we loop until n
            
        return centers[:n], radii[:n]

    def init_random(n):
        centers = []
        radii = []
        for _ in range(n):
            centers.append([np.random.rand(), np.random.rand()])
            radii.append(0.05)
        return centers, radii

    best_result = None
    best_sum = -1.0
    
    # Try multiple initializations
    inits = [init_grid_perturbed(n_circles), init_hexagonal(n_circles), init_random(n_circles)]
    # Add a few more random seeds
    for _ in range(3):
        inits.append(init_random(n_circles))

    for centers, radii in inits:
        vars0 = np.zeros(3 * n_circles)
        for i in range(n_circles):
            vars0[3*i] = centers[i][0]
            vars0[3*i+1] = centers[i][1]
            vars0[3*i+2] = radii[i]
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n_circles):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
        
        cons = get_constraints_dict(n_circles)
        
        try:
            # SLSQP can be slow with many constraints. 
            # We might need to reduce the number of active constraints or use a simpler approach.
            # However, with n=26, it's manageable.
            res = opt.minimize(lambda v: -np.sum(v[2::3]), # Objective: -sum(r)
                               vars0,
                               method='SLSQP',
                               bounds=bounds,
                               constraints=cons,
                               options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                final_radii = res.x[2::3]
                final_sum = np.sum(final_radii)
                if final_sum > best_sum:
                    best_sum = final_sum
                    best_result = res.x
        except Exception as e:
            print(f"Optimization failed: {e}")

    # Fallback or refinement if optimization didn't run well or returned bad result
    # If best_result is None or sum is low, we can try a force-based expansion
    
    if best_result is None or best_sum < 2.5:
        # Fallback to a simple force-based expansion
        centers = np.random.rand(n_circles, 2)
        radii = np.ones(n_circles) * 0.01
        
        # Simple iterative expansion
        for _ in range(1000):
            # Increase radii
            for i in range(n_circles):
                radii[i] += 0.0001
            
            # Resolve overlaps and boundaries
            changed = True
            iterations = 0
            while changed and iterations < 50:
                changed = False
                iterations += 1
                for i in range(n_circles):
                    # Check boundary
                    if centers[i, 0] - radii[i] < 0:
                        centers[i, 0] = radii[i]
                        changed = True
                    if centers[i, 0] + radii[i] > 1:
                        centers[i, 0] = 1 - radii[i]
                        changed = True
                    if centers[i, 1] - radii[i] < 0:
                        centers[i, 1] = radii[i]
                        changed = True
                    if centers[i, 1] + radii[i] > 1:
                        centers[i, 1] = 1 - radii[i]
                        changed = True
                    
                    # Check overlaps
                    for j in range(i + 1, n_circles):
                        dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                        min_dist = radii[i] + radii[j]
                        if dist < min_dist and dist > 0:
                            # Push apart
                            overlap = min_dist - dist
                            push_x = (centers[j, 0] - centers[i, 0]) / dist * overlap / 2
                            push_y = (centers[j, 1] - centers[i, 1]) / dist * overlap / 2
                            centers[i, 0] -= push_x
                            centers[i, 1] -= push_y
                            centers[j, 0] += push_x
                            centers[j, 1] += push_y
                            changed = True
                        elif dist == 0:
                            # Random push
                            centers[i, 0] += np.random.uniform(-0.01, 0.01)
                            centers[i, 1] += np.random.uniform(-0.01, 0.01)
                            centers[j, 0] += np.random.uniform(-0.01, 0.01)
                            centers[j, 1] += np.random.uniform(-0.01, 0.01)
                            changed = True
                
                # Clip centers to valid range for radii (roughly)
                for i in range(n_circles):
                    centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
                    centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])
                    
                    # If radius is too big for current position, reduce it
                    max_r_x = min(centers[i, 0], 1 - centers[i, 0])
                    max_r_y = min(centers[i, 1], 1 - centers[i, 1])
                    if radii[i] > max_r_x or radii[i] > max_r_y:
                        radii[i] = min(max_r_x, max_r_y)
                        changed = True

        final_centers = centers
        final_radii = radii
        best_sum = np.sum(final_radii)
        best_result = np.zeros(3 * n_circles)
        for i in range(n_circles):
            best_result[3*i] = final_centers[i, 0]
            best_result[3*i+1] = final_centers[i, 1]
            best_result[3*i+2] = final_radii[i]

    # Extract final result
    final_centers = np.array([[best_result[3*i], best_result[3*i+1]] for i in range(n_circles)])
    final_radii = np.array([best_result[3*i+2] for i in range(n_circles)])
    final_sum = np.sum(final_radii)
    
    # Final validation check inside code to ensure correctness
    # (The validate_packing function is provided but we should be safe)
    
    return final_centers, final_radii, final_sum

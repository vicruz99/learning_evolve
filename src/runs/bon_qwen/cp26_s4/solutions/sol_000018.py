# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 04e92922) state=463f4b5e sum of radii=0.426597 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing():
    N = 26
    # Initialize centers with a hexagonal-like lattice perturbed by randomness
    # We want to pack 26 circles. A 5x5 grid is 25. 
    # Let's try a hexagonal arrangement which is denser.
    
    # Initial guess: Random positions with small radii
    centers = np.random.uniform(0.1, 0.9, size=(N, 2))
    radii = np.full(N, 0.05) # Start with small valid radii
    
    # Helper to calculate constraints violations
    def get_violations(c, r):
        violations = []
        # Boundary violations
        for i in range(N):
            x, y = c[i]
            rad = r[i]
            if x - rad < 0: violations.append(rad - x)
            if x + rad > 1: violations.append(x + rad - 1)
            if y - rad < 0: violations.append(rad - y)
            if y + rad > 1: violations.append(y + rad - 1)
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                if dist < r[i] + r[j]:
                    violations.append(r[i] + r[j] - dist)
        return violations

    # Optimization loop
    # We will try to maximize sum(r) by iteratively growing r and fixing overlaps
    
    # Step 1: Initial layout improvement
    # Use a force-directed layout to spread circles out
    for _ in range(1000):
        # Calculate forces
        forces = np.zeros((N, 2))
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist == 0: dist = 1e-9
                # Desired distance is sum of radii + margin
                desired_dist = radii[i] + radii[j] + 0.02 # Keep them slightly apart to allow growth
                if dist < desired_dist:
                    repulsion = (desired_dist - dist) * 0.1 # Force strength
                    direction = diff / dist
                    forces[i] += direction * repulsion
                    forces[j] -= direction * repulsion
            
            # Boundary forces
            x, y = centers[i]
            r = radii[i]
            if x - r < 0.05: forces[i, 0] += (0.05 - (x-r)) * 0.5
            if x + r > 0.95: forces[i, 0] -= ((x+r) - 0.95) * 0.5
            if y - r < 0.05: forces[i, 1] += (0.05 - (y-r)) * 0.5
            if y + r > 0.95: forces[i, 1] -= ((y+r) - 0.95) * 0.5
        
        centers += forces
        # Clamp centers
        centers = np.clip(centers, 0, 1)

    # Step 2: Grow radii and optimize positions
    # We define an objective function to minimize overlap and boundary penetration
    # while maximizing sum of radii.
    # However, direct optimization of sum(r) with non-convex constraints is hard.
    # Instead, we can fix a target sum S and check feasibility, or iteratively expand.
    
    # Let's use scipy.optimize to find a local optimum for a fixed number of iterations
    # We will optimize both centers and radii.
    
    def objective(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        c = vars.reshape(N, 3) # (x, y, r)
        penalty = 0.0
        sum_r = 0.0
        
        # Boundary penalties
        for i in range(N):
            x, y, r = c[i]
            if r < 0: r = 0 # Ensure non-negative
            # Penalty for being outside
            if x - r < 0: penalty += 100 * (x - r)**2
            if x + r > 1: penalty += 100 * (x + r - 1)**2
            if y - r < 0: penalty += 100 * (y - r)**2
            if y + r > 1: penalty += 100 * (y + r - 1)**2
            sum_r += r
            
        # Overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((c[i, :2] - c[j, :2])**2))
                req_dist = c[i, 2] + c[j, 2]
                if dist < req_dist:
                    penalty += 100 * (req_dist - dist)**2
        
        # We want to maximize sum_r, so minimize -sum_r + penalty
        return -sum_r + penalty

    # Initial variables
    x0 = np.column_stack([centers, radii]).flatten()
    
    # Bounds for optimization
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)]) # x, y, r
        
    # Use SLSQP which handles bounds and constraints (implicitly via penalty here)
    # Or use trust-constr? SLSQP is often robust for this size.
    # Since constraints are handled by penalty in objective, we just minimize.
    # But penalty method can be unstable.
    # Let's use a constrained optimizer directly.
    
    # Defining constraints for SLSQP
    cons = []
    
    # 1. Boundary constraints: x - r >= 0, x + r <= 1, etc.
    for i in range(N):
        idx_x = i * 3
        idx_y = i * 3 + 1
        idx_r = i * 3 + 2
        
        # x - r >= 0  =>  r - x <= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_x] - v[idx_r]}) 
        # x + r <= 1  =>  x + r - 1 <= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v[idx_x] + v[idx_r])})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_y] - v[idx_r]})
        # y + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v[idx_y] + v[idx_r])})

    # 2. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # => (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_xi = i * 3
            idx_yi = i * 3 + 1
            idx_ri = i * 3 + 2
            idx_xj = j * 3
            idx_yj = j * 3 + 1
            idx_rj = j * 3 + 2
            
            def dist_constraint(v, i=i, j=j):
                xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
                xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                rad_sum_sq = (ri + rj)**2
                return dist_sq - rad_sum_sq
            
            cons.append({'type': 'ineq', 'fun': dist_constraint})

    # Objective: Maximize sum of radii => Minimize -sum(radii)
    def obj_func(v):
        s = 0
        for i in range(N):
            s += v[i * 3 + 2]
        return -s

    # Optimization
    # We run multiple restarts to avoid local minima
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Restart 1: From current configuration
    try:
        res = scipy.optimize.minimize(
            obj_func, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        if res.success or res.fun < best_sum * -1: # res.fun is negative sum
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(N, 3)[:, :2]
                best_radii = res.x.reshape(N, 3)[:, 2]
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Restart 2: Random initialization
    for _ in range(5):
        x0_rand = np.random.uniform(0.1, 0.9, size=(N, 3))
        x0_rand[:, 2] = np.random.uniform(0.02, 0.08, size=N) # Initial radii
        
        # Project to valid region roughly
        for i in range(N):
            x0_rand[i, 2] = min(x0_rand[i, 0], 1-x0_rand[i, 0], x0_rand[i, 1], 1-x0_rand[i, 1], x0_rand[i, 2])
            
        try:
            res = scipy.optimize.minimize(
                obj_func, x0_rand.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(N, 3)[:, :2]
                best_radii = res.x.reshape(N, 3)[:, 2]
        except:
            pass

    # Restart 3: Hexagonal Grid Initialization
    # Try to fit 26 circles in a hexagonal pattern
    # Estimate radius for hexagonal packing of 26 circles
    # Approximate r ~ 0.1
    # Let's generate coordinates
    hex_centers = []
    hex_radii = []
    r_est = 0.095
    # 5 rows
    # Row 0: 6 circles? Width 1. 6 circles need width ~ 1.1 * 2r?
    # Let's try a dense packing
    # 5, 6, 5, 6, 4 -> 26
    
    # Coordinates for hexagonal lattice
    # Vertical spacing sqrt(3)/2 * 2r = r*sqrt(3)
    # Horizontal spacing 2r
    # Shift odd rows by r
    
    rows = [5, 6, 5, 6, 4] # Sum = 26
    current_y = r_est
    dy = r_est * math.sqrt(3)
    
    temp_centers = []
    for k, count in enumerate(rows):
        row_x_start = r_est if k % 2 == 0 else 2 * r_est # Shift?
        # Actually standard hex: even rows start at r, odd rows start at 2r (shift r)
        # But we need to fit in [0,1].
        # If count=6, width needed: 2r + 5*(2r) = 12r? No.
        # Centers: r, 3r, 5r, 7r, 9r, 11r.
        # Rightmost center 11r. Must be <= 1-r => 12r <= 1 => r <= 0.0833.
        # If count=5, centers: r, 3r, 5r, 7r, 9r.
        # Rightmost 9r. <= 1-r => 10r <= 1 => r <= 0.1.
        # So mixed rows allow r up to 0.1 if no row has 6.
        # But we have rows with 6.
        # Can we shift differently?
        # If row has 6 circles, we need width.
        # Maybe stagger more?
        
        # Let's just generate points and let optimizer fix
        start_x = r_est + (0 if k % 2 == 0 else r_est) # Shift by r
        for i in range(count):
            x = start_x + i * 2 * r_est
            y = current_y
            if x <= 1 - r_est:
                temp_centers.append([x, y])
                hex_radii.append(r_est)
        current_y += dy
        
    # If we have less than 26, adjust or add
    # The above logic might produce < 26 if r_est is large.
    # Let's just use the optimized result from SLSQP if it's good.
    
    # Verify and refine the best result found
    if best_centers is not None:
        centers = best_centers
        radii = best_radii
        
        # One final polish: Optimize only positions with fixed radii to ensure no overlaps
        # Or just check validity.
        if validate_packing(centers, radii):
            return centers, radii, float(np.sum(radii))
        else:
            # If validation fails, try to repair
            # This might happen if SLSQP tolerance was loose
            # Re-run optimization with stricter bounds or repair manually
            pass

    # Fallback: Grid packing
    # 5x5 grid is 25 circles r=0.1. Sum=2.5.
    # We need 26.
    # Let's try to place 26 circles in a slightly distorted grid.
    centers = np.zeros((26, 2))
    radii = np.full(26, 0.095) # Slightly smaller to fit 26th
    
    # 5x5 grid points
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append((0.1 + 0.2*i, 0.1 + 0.2*j))
            
    # We have 25 points. Need 1 more.
    # Place in center? (0.5, 0.5) is occupied.
    # Place near center?
    # Or distort the grid.
    # Let's just use the optimized result if valid, else return a valid grid-based one.
    
    # Let's try a specific construction for 26
    # 6 rows?
    # Maybe 4, 5, 5, 5, 4, 3? Sum = 26.
    # Let's rely on the optimizer result.
    
    # If the optimizer didn't find a valid one, we construct a safe one.
    # 5x5 grid with r=0.095 is valid for 25 circles.
    # We can add a 26th small circle?
    # Where? In the gaps.
    # Gap between (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3).
    # Center of gap is (0.2, 0.2).
    # Distance to neighbors: sqrt(0.1^2 + 0.1^2) = 0.1414.
    # Sum of radii: 0.095 + r_new <= 0.1414 => r_new <= 0.0464.
    # Boundary check for (0.2, 0.2) with r=0.046: dist to wall 0.2. OK.
    # So we can add a small circle.
    # Sum = 25 * 0.095 + 0.046 = 2.375 + 0.046 = 2.42.
    # Still low.
    
    # The optimizer should find better.
    # Let's ensure we return a valid packing.
    
    # Re-attempt optimization with the grid initialization as seed if best is invalid
    if best_centers is None or not validate_packing(best_centers, best_radii):
        # Fallback to a known good structure
        # 5x5 grid with small perturbation to fit 26th
        centers = []
        radii = []
        # Let's try to place 26 circles using the optimizer result if it was close
        # Or just use a dense hexagonal packing found by literature/heuristics.
        
        # Let's run a quick simulation to improve the grid
        # 5x5 grid
        pts = []
        for r in range(5):
            for c in range(5):
                pts.append([0.1 + 0.2*c, 0.1 + 0.2*r])
        # Add one in center? No, occupied.
        # Add one at (0.5, 0.5)? Occupied.
        # Maybe shift everything to make space?
        
        # Let's use the optimizer's last best valid result if any, 
        # otherwise construct a valid one.
        
        # Let's assume the optimizer worked.
        # If not, let's force a valid solution.
        # We can take the best result and project it to valid.
        
        # Simple projection:
        # If overlap, shrink radii until valid.
        # This guarantees validity but lowers sum.
        
        centers = best_centers if best_centers is not None else np.random.rand(26, 2)
        radii = best_radii if best_radii is not None else np.full(26, 0.01)
        
        # Projection loop
        for _ in range(100):
            valid = True
            # Check overlaps
            for i in range(26):
                for j in range(i+1, 26):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < radii[i] + radii[j]:
                        # Shrink both
                        overlap = (radii[i] + radii[j]) - dist
                        radii[i] -= overlap / 2
                        radii[j] -= overlap / 2
                        valid = False
            # Check boundaries
            for i in range(26):
                x, y = centers[i]
                r = radii[i]
                if x - r < 0: radii[i] = x
                if x + r > 1: radii[i] = 1 - x
                if y - r < 0: radii[i] = y
                if y + r > 1: radii[i] = 1 - y
                if radii[i] < 0: radii[i] = 0
            if valid: break
            
        # This might result in very small radii if configuration is bad.
        # But we had an optimizer run.
        
        # Let's return the optimizer result if it seems reasonable (sum > 2.0)
        if np.sum(radii) > 2.0:
             return centers, radii, float(np.sum(radii))
        
        # If sum is too low, it means optimizer failed or projection killed it.
        # Let's try to return a hardcoded valid packing if needed?
        # But the optimizer should work.
        
    return centers, radii, float(np.sum(radii))

# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4705e2a5) state=ec3959f0 sum of radii=1.335346 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing():
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    n = 26
    
    # 1. Initialization: Hexagonal lattice packing
    # We want to place 26 points.
    # A hexagonal grid is denser.
    # Let's try rows with counts like 6, 5, 6, 5, 4? Sum = 26.
    # Or 5, 6, 5, 6, 4?
    # Let's try to fit them in a square.
    # Width of k circles in hexagonal row is 2*r*k? No, span is 2r(k-1) + 2r = 2rk?
    # Centers at r, 3r... width constraint 2rk <= 1 => r <= 1/(2k).
    # If max k=6, r <= 1/12 = 0.0833.
    # If max k=5, r <= 0.1.
    # To maximize sum, we prefer larger r. So we should avoid rows of 6 if possible.
    # But 26 = 5*5 + 1. We can't do all rows of 5.
    # Maybe 5, 5, 5, 5, 6? One row of 6.
    # Or maybe we don't need a perfect lattice.
    
    # Let's start with a perturbed grid or random points and let optimizer work.
    # A good start is a grid of 26 points.
    # 26 points in a 5x5 grid has 1 overlap.
    # Let's just place them randomly but with some spacing.
    
    np.random.seed(42)
    
    # Create a 5x5 grid plus one point
    # Grid points
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + i*0.2, 0.1 + j*0.2])
    # Add 26th point in a gap? 
    # The grid is full.
    # Let's just scatter 26 points nicely.
    # Use a hexagonal packing initialization logic.
    # Approximate radius 0.1. Spacing 0.2.
    
    # Let's try a simple hexagonal pattern generator
    init_centers = []
    r_est = 0.09 # slightly less than 0.1 to fit 26
    dy = math.sqrt(3) * r_est
    dx = 2 * r_est
    
    rows = 6
    # Try to fill rows
    count = 0
    y = r_est
    row_idx = 0
    while count < n and y + r_est <= 1.0:
        # Determine shift for this row
        shift = dx / 2 if row_idx % 2 == 1 else 0
        x = r_est + shift
        # Check if first circle fits
        while x + r_est <= 1.0 and count < n:
            init_centers.append([x, y])
            x += dx
            count += 1
        y += dy
        row_idx += 1
    
    # If we didn't get 26 points, fallback to random or grid
    if len(init_centers) < n:
        # Fallback: Random uniform
        init_centers = np.random.rand(n, 2)
    else:
        init_centers = np.array(init_centers[:n])

    # Initial radii
    radii = np.full(n, 0.01)
    
    # Variables: x_0, y_0, r_0, ..., x_25, y_25, r_25
    # Total 3 * 26 = 78 variables.
    # Flattened array.
    
    def objective(vars):
        # Maximize sum of radii => minimize negative sum
        r = vars[2::3]
        return -np.sum(r)
    
    def gradient(vars):
        # Gradient of -sum(r) is -1 for r components, 0 for x,y
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        return grad

    # Constraints
    # 1. Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # 2. Pairwise distance: dist >= r_i + r_j => dist^2 >= (r_i + r_j)^2
    #    Or dist - r_i - r_j >= 0
    
    # To avoid sqrt in constraint evaluation (speed), we can use squared distance,
    # but r_i + r_j is not squared nicely.
    # Let's use the linear constraint: sqrt(...) - r_i - r_j >= 0.
    # It is smooth enough.
    
    # We will use a penalty method inside the objective or use constraints in minimize.
    # Using 'SLSQP' with constraints.
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx] - v[3*idx + 2]
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx + 2]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx + 1] - v[3*idx + 2]
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[3*idx + 1] - v[3*idx + 2]
        })
        
    # Pairwise constraints
    # dist(c_i, c_j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) - v[3*i+2] - v[3*j+2]
            })
            
    # Bounds
    # x, y in [0, 1]
    # r >= 0 (and implicitly <= 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Initial vector
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = init_centers[i, 0]
        x0[3*i+1] = init_centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Optimization
    # SLSQP might be slow with 400+ constraints.
    # Let's try a simpler approach: 
    # Iteratively increase radii and fix overlaps?
    # Or just run SLSQP and hope it converges.
    # Given the time limit, maybe a few iterations of a custom solver is safer.
    
    # Let's try a custom iterative solver (gradient ascent with projection/repulsion)
    # This is often more robust for packing problems.
    
    centers = np.array(init_centers)
    current_radii = np.full(n, 0.05) # Start with reasonable radii
    
    # We want to maximize sum(r).
    # Let's perform steps:
    # 1. Try to increase all radii by a small amount.
    # 2. If overlaps occur, move centers to resolve them.
    # 3. Repeat until convergence.
    
    # This is similar to a simulated annealing or force-directed layout.
    
    best_centers = centers.copy()
    best_radii = current_radii.copy()
    best_sum = np.sum(best_radii)
    
    # Learning rate for radius increase
    delta_r = 0.001
    # Learning rate for position update
    lr = 0.01
    
    # Number of iterations
    iters = 2000
    
    for step in range(iters):
        # Attempt to increase radii
        # We can increase radii greedily.
        # If we increase radii, we might violate constraints.
        # We will compute repulsive forces based on overlaps.
        
        # Current radii increase
        # To maximize sum, we want to push radii up.
        # Let's just increase them by a factor or fixed amount.
        # But we need to balance.
        # A stable way: 
        # Compute max possible radius for each circle given current centers and other radii?
        # No, it's coupled.
        
        # Let's use a simple repulsion force.
        # Force on i due to j: if dist < r_i + r_j, push apart.
        # Force magnitude proportional to overlap.
        # Also boundary repulsion.
        
        forces = np.zeros((n, 2))
        
        # Pairwise interactions
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                overlap = current_radii[i] + current_radii[j] - dist
                
                if overlap > 0:
                    # Repulsive force
                    if dist > 1e-9:
                        direction = diff / dist
                    else:
                        direction = np.random.rand(2) * 2 - 1 # Random if too close
                    
                    # Force proportional to overlap
                    # We want to push centers apart to allow radii to grow.
                    # Or we can fix radii and move centers.
                    # Here we are doing both?
                    # Let's just move centers to resolve overlaps, and try to grow radii.
                    
                    force_mag = overlap * 10.0 # Stiff spring
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
        
        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = current_radii[i]
            
            # Left
            if x - r < 0:
                forces[i, 0] += (r - x) * 10.0
            # Right
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 10.0
            # Bottom
            if y - r < 0:
                forces[i, 1] += (r - y) * 10.0
            # Top
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 10.0
                
        # Update centers
        # We want to increase radii too.
        # If forces are small (no overlap), we can increase radii.
        max_force = np.max(np.abs(forces))
        
        if max_force < 1e-6:
            # No significant overlap, grow radii
            current_radii += delta_r
            # Cap radii? No, boundaries will handle it.
        else:
            # Resolve overlaps by moving centers
            centers += forces * lr
            # Clamp centers to [0, 1]
            centers = np.clip(centers, 0, 1)
            
        # Update best if sum increases and valid?
        # Checking validity every step is expensive.
        # Let's check periodically.
        
        if step % 100 == 0:
            # Check validity roughly
            # If valid, record
            valid = True
            # Check boundaries
            for i in range(n):
                if current_radii[i] < 0: valid = False
                if centers[i,0] < current_radii[i] - 1e-9 or centers[i,0] > 1 - current_radii[i] + 1e-9: valid = False
                if centers[i,1] < current_radii[i] - 1e-9 or centers[i,1] > 1 - current_radii[i] + 1e-9: valid = False
            if not valid:
                # If invalid, shrink radii slightly to recover
                current_radii *= 0.9
                continue

            # Check overlaps
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < current_radii[i] + current_radii[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                s = np.sum(current_radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers.copy()
                    best_radii = current_radii.copy()
            
            # Decay learning rate
            lr *= 0.99
            delta_r *= 0.99
            
    # Final optimization step using scipy to polish
    # Use the best configuration found so far
    x0_polish = np.zeros(3 * n)
    for i in range(n):
        x0_polish[3*i] = best_centers[i, 0]
        x0_polish[3*i+1] = best_centers[i, 1]
        x0_polish[3*i+2] = best_radii[i]
        
    try:
        res = opt.minimize(objective, x0_polish, method='SLSQP', jac=gradient, bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-12})
        
        # Extract results
        c_opt = np.zeros((n, 2))
        r_opt = np.zeros(n)
        for i in range(n):
            c_opt[i, 0] = res.x[3*i]
            c_opt[i, 1] = res.x[3*i+1]
            r_opt[i] = res.x[3*i+2]
            
        # Validate
        if validate_packing(c_opt, r_opt):
            return c_opt, r_opt, np.sum(r_opt)
    except Exception:
        pass

    # If polish fails or we didn't find a valid one during heuristic, return best heuristic result
    # Ensure validity
    # The heuristic might have slight errors.
    # Let's run a validation check on best_radii/centers
    # If invalid, shrink radii until valid.
    
    # Shrink radii to ensure validity
    while True:
        valid = True
        # Check boundaries
        for i in range(n):
            if best_radii[i] < 0: valid = False; break
            if best_centers[i,0] < best_radii[i] - 1e-9 or best_centers[i,0] > 1 - best_radii[i] + 1e-9: valid = False; break
            if best_centers[i,1] < best_radii[i] - 1e-9 or best_centers[i,1] > 1 - best_radii[i] + 1e-9: valid = False; break
        if not valid:
            best_radii *= 0.95
            continue
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < best_radii[i] + best_radii[j] - 1e-9:
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            best_radii *= 0.95
        else:
            break
            
    return best_centers, best_radii, np.sum(best_radii)

# We need to define validate_packing inside or assume it's available?
# The prompt says "We will run the below validation function".
# So I should not redefine it, but I can use it?
# Actually, I can't import it.
# But I can copy its logic for my own checks if needed, or just rely on the logic I wrote.
# To be safe, I'll include a local check or just trust my logic.
# The prompt implies the validator is external.
# But I need to return valid data.
# I will implement a local check helper to be sure.

def check_validity(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any(): return False
    for i in range(n):
        if radii[i] < 0: return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12: return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12: return False
    return True

# Redefine run_packing to use the helper and ensure validity
def run_packing():
    n = 26
    np.random.seed(123) # Fixed seed for reproducibility
    
    # Initialization: Hexagonal grid
    # 26 points.
    # 6 rows?
    # Let's try to generate points manually to ensure good spread.
    # A 5x5 grid is 25 points.
    # Let's perturb it.
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + i*0.2, 0.1 + j*0.2])
    # Add one point in the center? (0.5, 0.5) is occupied.
    # Maybe (0.5, 0.3) ?
    # Actually, just use a dense random start.
    pts = np.random.rand(n, 2)
    
    # Better initialization:
    # Use the grid but shrink it to allow 26th.
    # 5x5 grid radius 0.1.
    # If we shrink to 0.09, we have space.
    # Let's initialize with 5x5 grid + 1 point.
    grid = []
    for r in range(5):
        for c in range(5):
            grid.append([0.1 + c*0.2, 0.1 + r*0.2])
    # 25 points. Add one at (0.5, 0.5) but it's at (0.9, 0.9) in 0-index?
    # (0.5, 0.5) is the center of the square.
    # In grid, (0.1+2*0.2, 0.1+2*0.2) = (0.5, 0.5).
    # So grid covers center.
    # Let's just use random.
    centers = np.random.rand(n, 2) * 0.8 + 0.1 # Keep away from edges initially
    
    radii = np.full(n, 0.02)
    
    # Optimization loop
    lr_pos = 0.01
    lr_rad = 0.0005
    
    for _ in range(3000):
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                target_dist = radii[i] + radii[j]
                
                if dist < 1e-9:
                    dist = 1e-9
                    vec = np.random.rand(2)
                
                # Overlap amount
                overlap = target_dist - dist
                
                if overlap > 0:
                    # Force to push apart
                    # Normalize vector
                    norm_vec = vec / dist
                    # Force magnitude
                    f = overlap * 20.0 # Strong repulsion
                    forces[i] += norm_vec * f
                    forces[j] -= norm_vec * f
                else:
                    # Weak attraction? No, we want to expand.
                    # Maybe slight attraction to keep them clustered? 
                    # Not necessary.
                    pass

        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # x - r >= 0 => if x < r, push right
            if x < r:
                forces[i, 0] += (r - x) * 20.0
            # x + r <= 1 => if x > 1 - r, push left
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 20.0
            # y - r >= 0
            if y < r:
                forces[i, 1] += (r - y) * 20.0
            # y + r <= 1
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 20.0
        
        # Update positions
        centers += forces * lr_pos
        centers = np.clip(centers, 0, 1)
        
        # Try to increase radii
        # Check if we can increase.
        # A simple heuristic: if max force is small, increase radii.
        max_f = np.max(np.abs(forces))
        if max_f < 1e-4:
            radii += lr_rad
        else:
            # If overlaps, shrink radii slightly to stabilize
            radii *= 0.999
            
        # Decay rates
        lr_pos *= 0.9995
        lr_rad *= 0.9995
        
    # Final validation and shrinking
    # Ensure strict validity
    # The simulation might have small errors.
    # Run a few correction steps.
    for _ in range(100):
        valid = True
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < radii[i] + radii[j] - 1e-12:
                    # Shrink radii to fix
                    avg_r = (radii[i] + radii[j]) / 2
                    needed = d / 2
                    factor = needed / avg_r
                    radii[i] *= factor * 0.99
                    radii[j] *= factor * 0.99
                    valid = False
        # Check boundaries
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            min_dist = min(x, 1-x, y, 1-y)
            if r > min_dist + 1e-12:
                radii[i] = min_dist * 0.99
                valid = False
        if valid:
            break
            
    # One last check with the provided logic (simulated)
    if not check_validity(centers, radii):
        # If still invalid, reduce all radii significantly
        scale = 0.9
        while not check_validity(centers, radii):
            radii *= scale
            scale *= 0.9
            if scale < 0.001: break # Safety
            
    return centers, radii, np.sum(radii)

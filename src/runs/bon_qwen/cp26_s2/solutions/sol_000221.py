# sol_000221 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=d071d918 sum of radii=2.615130 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def get_initial_config(n=26):
    """
    Generates an initial configuration of n circles.
    Uses a perturbed grid/hexagonal layout.
    """
    # Try to arrange in a rectangular grid roughly sqrt(n) x sqrt(n)
    # 26 is close to 5x5=25. A 5x6 grid has 30 spots.
    # We can try to fit them in a pattern.
    
    # Let's try a hexagonal packing initialization which is denser.
    # Estimate radius ~ 0.1
    r_est = 0.1
    centers = []
    
    # Rows in hex packing
    # Vertical spacing = r * sqrt(3)
    # Horizontal spacing = 2r
    # We need to fit 26 circles.
    
    # Let's just place them in a grid and perturb
    # 5 rows, 6 cols = 30 points. We take 26.
    # Grid size: 0.2 step.
    x_coords = np.linspace(0.1, 0.9, 5) # 5 cols? No, 0.1 to 0.9 is 5 points.
    # 5 points * 5 points = 25. Need 26.
    # Let's use 6 columns and 5 rows, step 1/7 approx?
    # Or just random good start.
    
    # Better: Fill a grid
    # 5 rows, 6 columns = 30 slots.
    # Select 26 best? Or just random 26.
    # Let's use a deterministic subset.
    
    slots = []
    # 5 rows
    for r in range(5):
        y = 0.1 + r * 0.2 # 0.1, 0.3, 0.5, 0.7, 0.9
        # 6 cols
        for c in range(6):
            x = 0.08333 + c * 0.16666 # roughly 1/12, 3/12...
            # Actually simpler: uniform grid
            x = (2*c + 1) / 12.0 
            slots.append([x, y])
            
    # slots has 30 points. We need 26.
    # Keep first 26.
    init_centers = np.array(slots[:26])
    
    # Add some random jitter to break symmetry
    np.random.seed(42)
    jitter = np.random.uniform(-0.02, 0.02, size=init_centers.shape)
    init_centers += jitter
    init_centers = np.clip(init_centers, 0.05, 0.95)
    
    # Initial radii: assume equal, slightly small to allow optimization
    # If we start with r=0.1, constraints might be violated immediately.
    # Start with r=0.05 and let optimizer grow them.
    init_radii = np.full(n, 0.05)
    
    return init_centers, init_radii

def validate_and_refine(centers, radii):
    """
    Ensures radii are valid and non-negative.
    """
    radii = np.maximum(radii, 0.0)
    return centers, radii

def objective(vars, n):
    """
    Objective function: Maximize sum of radii (minimize negative sum).
    vars layout: [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = vars[2::3]
    return -np.sum(radii)

def constraints_factory(centers, radii, n):
    """
    This function is not used directly in scipy with vectorized vars 
    unless we structure vars carefully.
    """
    pass

def run_packing():
    n = 26
    
    # 1. Initialization
    # We use a grid initialization but optimized coordinates
    # Let's try a hexagonal-like arrangement for better packing
    centers, radii = get_initial_config(n)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    # Actually, better to optimize [x1, y1, ..., x26, y26, r] (equal radii) first?
    # Or optimize all. Let's try optimizing all.
    # Variables: 26*3 = 78 variables.
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds for variables
    # x, y in [0, 1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 1.0)) # r (upper bound 1 is loose)
        
    # Constraints
    # 1. Boundary: r_i <= x_i, x_i + r_i <= 1, r_i <= y_i, y_i + r_i <= 1
    #    => x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    # 2. Non-overlap: dist(i,j) >= r_i + r_j
    #    => dist(i,j)^2 - (r_i + r_j)^2 >= 0
    
    cons = []
    
    # Helper to extract vars
    # x0[3*i], x0[3*i+1], x0[3*i+2]
    
    # Boundary constraints
    for i in range(n):
        # x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2]
        })
        # 1 - x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]
        })
        # y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
        })
        # 1 - y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]
        })
        
    # Overlap constraints
    # For i < j: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    # This is non-convex. SLSQP might struggle.
    # However, with good initialization it should work.
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: \
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # Optimization
    # Using SLSQP
    # Might need multiple restarts or a robust method.
    # Let's try one run with good init.
    
    try:
        res = opt.minimize(
            fun=lambda v: objective(v, n),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if res.success:
            x0 = res.x
    except Exception:
        pass

    # Extract result
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i] = [x0[3*i], x0[3*i+1]]
        final_radii[i] = x0[3*i+2]
        
    # Validate and fix any tiny violations due to numerical error
    # Check radii
    final_radii = np.maximum(final_radii, 0.0)
    
    # Check overlaps and reduce radii if necessary (simple fix)
    # If optimization failed to satisfy constraints perfectly, we might need to scale down.
    # But let's assume SLSQP did its job.
    
    # Just to be safe, let's run a quick check and shrink radii if overlapping
    # This is a fallback
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(final_centers[i] - final_centers[j])
            min_dist = final_radii[i] + final_radii[j]
            if dist < min_dist:
                # Reduce radii proportionally to distance
                scale = dist / min_dist
                final_radii[i] *= scale
                final_radii[j] *= scale
                # This is a heuristic fix, might need iteration
                valid = False
    
    if not valid:
        # If still issues, global scale down
        # Find max violation
        max_viol = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(final_centers[i] - final_centers[j])
                req = final_radii[i] + final_radii[j]
                if dist < req:
                    max_viol = max(max_viol, req - dist)
        
        if max_viol > 0:
            # Scale all radii down slightly?
            # Or just trust the optimizer. 
            # Let's re-run a simple scaling to ensure validity
            # Find lambda such that lambda*(r_i + r_j) <= dist for all i,j
            # lambda <= dist / (r_i + r_j)
            # lambda = min over all pairs of dist / (r_i + r_j)
            # Also boundary constraints: lambda*r_i <= x_i, etc.
            
            min_lambda = 1.0
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(final_centers[i] - final_centers[j])
                    denom = final_radii[i] + final_radii[j]
                    if denom > 1e-9:
                        min_lambda = min(min_lambda, dist / denom)
                
                # Boundaries
                r = final_radii[i]
                if r > 1e-9:
                    min_lambda = min(min_lambda, final_centers[i,0]/r)
                    min_lambda = min(min_lambda, (1-final_centers[i,0])/r)
                    min_lambda = min(min_lambda, final_centers[i,1]/r)
                    min_lambda = min(min_lambda, (1-final_centers[i,1])/r)
            
            # Apply lambda with small epsilon
            final_radii *= (min_lambda * 0.9999)
            
            # Also ensure centers are valid (they are in [0,1] by bounds)
            # But with reduced radii, boundaries are satisfied.

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Note: The problem statement asks to return the function run_packing.
# The code above defines it. 
# However, to be safe and ensure the code runs correctly within the provided environment,
# we should ensure imports are inside or available. 
# The prompt allows scipy.

# Let's refine the solution to be more robust. 
# The SLSQP might get stuck. 
# A force-directed relaxation is often more robust for packing.

def run_packing_force():
    n = 26
    # Initialization
    centers = np.zeros((n, 2))
    # Grid initialization
    # 5 rows, 6 cols grid logic
    idx = 0
    for r in range(5):
        y = 0.1 + r * 0.2
        for c in range(6):
            if idx < n:
                x = (2*c + 1) / 12.0
                centers[idx] = [x, y]
                idx += 1
    
    # Perturb
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(n, 0.05) # Start small
    
    # Simulation parameters
    dt = 0.01
    k_rep = 10.0
    k_boundary = 10.0
    r_target = 0.10 # Target radius to grow towards? No, we grow dynamically.
    
    # We will try to increase radius slowly while maintaining non-overlap
    # Actually, we can just run a simulation where circles repel each other and walls,
    # and we have a force trying to expand radii.
    
    current_r = 0.05
    step = 0.001
    
    # Number of iterations
    iterations = 5000
    
    for it in range(iterations):
        # Try to increase radius
        # Check if valid with current_r + step
        test_r = current_r + step
        valid_step = True
        
        # Check overlaps with test_r
        # This is O(N^2), N=26 is small.
        # But doing this every iteration is slow? 5000 * 300 ops is fine.
        
        # Actually, better to compute forces and move centers.
        
        forces = np.zeros_like(centers)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = 2 * current_r
                if dist < min_dist and dist > 1e-6:
                    # Overlap
                    repulsion = (min_dist - dist) / dist * diff
                    forces[i] += repulsion
                    forces[j] -= repulsion
                elif dist < min_dist * 1.5 and dist > 1e-6:
                    # Soft repulsion to keep spacing
                    repulsion = 1.0 / (dist * dist) * diff
                    forces[i] += repulsion
                    forces[j] -= repulsion
        
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            # Left wall
            if x - current_r < 0:
                forces[i, 0] += k_boundary * (current_r - x) # Push right
            # Right wall
            if x + current_r > 1:
                forces[i, 0] -= k_boundary * (x + current_r - 1) # Push left
            # Bottom wall
            if y - current_r < 0:
                forces[i, 1] += k_boundary * (current_r - y) # Push up
            # Top wall
            if y + current_r > 1:
                forces[i, 1] -= k_boundary * (y + current_r - 1) # Push down
            
            # Hard constraints: keep inside [0,1]
            # Project forces?
            # Just clamp position after move
        
        # Apply forces
        centers += forces * dt
        
        # Clamp centers
        centers = np.clip(centers, 0.0, 1.0)
        
        # If no overlaps, try increasing radius
        # Check overlaps
        overlaps = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 2 * current_r - 1e-7:
                    overlaps += 1
                # Check boundaries
                if centers[i,0] - current_r < -1e-7 or centers[i,0] + current_r > 1 + 1e-7 or \
                   centers[i,1] - current_r < -1e-7 or centers[i,1] + current_r > 1 + 1e-7:
                    overlaps += 1 # Count as overlap/invalid
        
        if overlaps == 0:
            current_r += step
        else:
            # If overlaps, maybe shrink radius slightly to recover?
            # Or just stay.
            # If stuck, maybe reduce step?
            pass
            
    radii = np.full(n, current_r)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Combining ideas: The force method is good for equal radii.
# For unequal, we need the LP approach or the optimizer.
# Given the constraints and time, the optimizer with SLSQP is the most "expert" way 
# to handle the general case, provided initialization is good.
# But SLSQP with 78 variables and 300+ constraints can be slow or fail.
# Let's stick to the force method but optimize for equal radii first, 
# then maybe perturb.
# Actually, the prompt allows different radii.
# But as discussed, equal radii sum is limited by ~2.6.
# The target 2.636 suggests unequal might be needed, but maybe my bound is loose.
# Let's provide the force-based equal radii solution, which is robust.
# If it hits 2.6, it's a good start.
# To beat 2.6, we might need to allow radii to differ.
# But implementing a robust unequal solver in one shot is risky.
# Let's try to maximize sum of radii using the force method but allowing radii to grow individually?
# That's complex.

# Let's refine the SLSQP approach with a better initialization.
# A hexagonal packing is better than square grid.

def run_packing():
    n = 26
    
    # Hexagonal initialization
    centers = np.zeros((n, 2))
    idx = 0
    
    # Estimate radius 0.1
    r_est = 0.1
    # Row height sqrt(3)/2 * 2r = r*sqrt(3) ~ 0.1732
    # Width 2r = 0.2
    
    # Let's pack rows
    y = r_est
    row = 0
    while idx < n:
        # Determine number of circles in this row
        # If row is even, start at r_est. If odd, start at r_est + r_est? 
        # Hexagonal: row 0: x=r, x=3r, ...
        # row 1: x=2r, x=4r, ...
        
        start_x = r_est if row % 2 == 0 else 2 * r_est
        x = start_x
        while x + r_est <= 1.0 and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r_est
        
        y += r_est * np.sqrt(3)
        row += 1
        
    # If we didn't fill 26, fill remaining randomly or extend
    if idx < n:
        # Fill remaining in a grid pattern
        for i in range(idx, n):
            # Random valid position
            cx, cy = np.random.uniform(0.1, 0.9, 2)
            centers[i] = [cx, cy]
            
    # Add small jitter
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # Initial radii small
    radii = np.full(n, 0.01)
    
    # Flatten
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 1)])
        
    cons = []
    # Boundary
    for i in range(n):
        # x >= r
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
    # Overlap
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: \
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })
            
    # Optimize
    # Use Powell or SLSQP. SLSQP is good for constraints.
    try:
        res = opt.minimize(
            fun=lambda v: -np.sum(v[2::3]),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-10}
        )
        x0 = res.x
    except:
        pass
        
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i] = [x0[3*i], x0[3*i+1]]
        final_radii[i] = max(0.0, x0[3*i+2])
        
    # Post-process to ensure strict validity
    # Scale down if necessary
    min_lambda = 1.0
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(final_centers[i] - final_centers[j])
            req = final_radii[i] + final_radii[j]
            if req > 1e-9:
                min_lambda = min(min_lambda, dist / req)
        
        r = final_radii[i]
        if r > 1e-9:
            min_lambda = min(min_lambda, final_centers[i,0]/r, (1-final_centers[i,0])/r)
            min_lambda = min(min_lambda, final_centers[i,1]/r, (1-final_centers[i,1])/r)
            
    if min_lambda < 1.0:
        final_radii *= min_lambda * 0.99999 # slightly smaller to be safe
        
    return final_centers, final_radii, np.sum(final_radii)

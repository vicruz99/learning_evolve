import numpy as np
from scipy.optimize import minimize

def compute_violations(centers, radii):
    """
    Computes the sum of squared violations for overlaps and boundary constraints.
    """
    n = centers.shape[0]
    violation = 0.0

    # Boundary violations: circles must be within [0, 1]
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # Same for y
    violation += np.sum(np.maximum(0, radii - centers[:, 0])**2)
    violation += np.sum(np.maximum(0, radii - (1 - centers[:, 0]))**2)
    violation += np.sum(np.maximum(0, radii - centers[:, 1])**2)
    violation += np.sum(np.maximum(0, radii - (1 - centers[:, 1]))**2)

    # Overlap violations: dist(i, j) >= r_i + r_j
    # We only check i < j
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                violation += (sum_r - dist)**2
                
    return violation

def objective_function(params, n):
    """
    Objective function to minimize: -sum(radii) + penalty * violations
    """
    # Unpack parameters
    # params layout: [x1, y1, r1, x2, y2, r2, ...]
    centers = params[0::3].reshape(n, 2)
    radii = params[1::3]
    
    # We pass penalty via a closure or global, but for simplicity 
    # we can compute violations here.
    # However, to make penalty adjustable, we might need a wrapper.
    # But for this implementation, we will use a fixed high penalty 
    # or implement a simple loop in run_packing.
    # Actually, let's just return the value.
    
    # To avoid recomputing everything in a complex way, 
    # we can assume the caller handles the penalty logic or we use a global.
    # But let's stick to the function signature.
    # We'll use a global penalty factor for this run.
    global PENALTY_FACTOR
    
    v = compute_violations(centers, radii)
    return -np.sum(radii) + PENALTY_FACTOR * v

def run_packing():
    """
    Main function to perform the packing optimization.
    """
    n_circles = 26
    
    # Global penalty factor
    global PENALTY_FACTOR
    
    # 1. Initialization
    # Place circles in a dense grid pattern to start close to a solution
    # A 5x5 grid is 25 circles. We can add one in the middle of a gap or perturb.
    # Let's use a hexagonal-ish layout or just a perturbed grid.
    
    # Initial layout: 5 rows, varying counts to fill 26
    # Rows: 5, 6, 5, 6, 4 = 26 circles.
    # This might be too wide for unit square if r is large, but good for start.
    
    centers = []
    # We'll try a 6x5 grid logic but remove some to get 26, or just random dense.
    # Random dense initialization with rejection sampling is safer to avoid immediate huge overlaps?
    # No, structured is better.
    
    # Let's place them in a 6x5 grid (30 spots) and keep first 26.
    # Grid spacing roughly 1/6.
    # Actually, let's just place them on a grid and optimize.
    
    x_coords = []
    y_coords = []
    # 5 columns, 6 rows? 5*6=30.
    # Let's do 6 rows, 5 cols.
    cols = 5
    rows = 6
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n_circles:
                break
            # Shift odd rows for hex packing
            y = (r + 0.5) / rows
            x = (c + 0.5) / cols
            if r % 2 == 1:
                x += 0.5 / cols # Shift
            # Clip to ensure inside
            x = np.clip(x, 0.1, 0.9)
            y = np.clip(y, 0.1, 0.9)
            centers.append([x, y])
            count += 1
        if count >= n_circles:
            break
            
    centers = np.array(centers)
    # Initial radii small
    radii = np.full(n_circles, 0.05)
    
    # Flatten to params
    params = []
    for i in range(n_circles):
        params.append(centers[i, 0])
        params.append(centers[i, 1])
        params.append(radii[i])
    params = np.array(params)
    
    # Bounds for optimization
    # x, y in [0, 1], r >= 0
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 1.0)) # r (upper bound 1 is loose)
        
    # 2. Optimization with increasing penalty (Continuation method)
    # This helps in finding a valid packing with large radii.
    
    PENALTY_FACTOR = 100.0
    
    # We need to wrap objective to use current PENALTY_FACTOR
    def obj_wrapper(p):
        return objective_function(p, n_circles)

    # Initial optimization run
    res = minimize(obj_wrapper, params, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
    params = res.x
    
    # Increase penalty and refine
    for step in range(5):
        PENALTY_FACTOR *= 10
        res = minimize(obj_wrapper, params, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
        params = res.x
        
        # Check if we are stuck or converged
        if res.success or res.nit == 0:
            pass # Continue increasing penalty anyway to tighten constraints
            
    # Final extraction
    final_centers = params[0::3].reshape(n_circles, 2)
    final_radii = params[1::3]
    
    # Clip radii to be non-negative (though optimizer should handle it)
    final_radii = np.maximum(final_radii, 0.0)
    
    # Clip centers to ensure they are valid given radii?
    # The optimizer minimizes violations, but might leave small violations.
    # We should project centers to valid region [r, 1-r]
    for i in range(n_circles):
        r = final_radii[i]
        x, y = final_centers[i]
        final_centers[i, 0] = np.clip(x, r, 1 - r)
        final_centers[i, 1] = np.clip(y, r, 1 - r)
        
    # Validate and adjust if necessary (simple projection for overlaps might be needed)
    # But L-BFGS with high penalty should have resolved most overlaps.
    # Just to be safe, run a quick local relaxation to fix any tiny overlaps.
    
    # Local relaxation (Force-directed)
    dt = 0.001
    stiffness = 100.0
    for _ in range(200):
        forces = np.zeros_like(final_centers)
        # Check overlaps
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_vec = final_centers[j] - final_centers[i]
                dist = np.linalg.norm(dist_vec)
                if dist < 1e-9: dist = 1e-9 # Avoid div by zero
                sum_r = final_radii[i] + final_radii[j]
                
                if dist < sum_r:
                    overlap = sum_r - dist
                    force_magnitude = overlap * stiffness
                    force_dir = dist_vec / dist
                    forces[i] -= force_dir * force_magnitude
                    forces[j] += force_dir * force_magnitude
        
        # Boundary forces
        for i in range(n_circles):
            r = final_radii[i]
            x, y = final_centers[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * stiffness
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * stiffness
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * stiffness
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * stiffness
                
        final_centers += forces * dt
        # Project back to valid box
        for i in range(n_circles):
            r = final_radii[i]
            final_centers[i, 0] = np.clip(final_centers[i, 0], r, 1 - r)
            final_centers[i, 1] = np.clip(final_centers[i, 1], r, 1 - r)

    # Calculate sum of radii
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, float(sum_radii)

# Global variable for closure
PENALTY_FACTOR = 0.0
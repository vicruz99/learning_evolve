# sol_000190 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 083f9270) state=d57384af sum of radii=1.606842 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    N = 26
    np.random.seed(42) # For reproducibility
    
    # 1. Initialization
    # We generate a hexagonal grid to start with a dense packing.
    # We need to select 26 points from a sufficiently dense lattice.
    
    # Let's create a grid of points
    # Approximate optimal radius r ~ 0.1. Spacing ~ 0.2.
    # 1 / 0.2 = 5. So 5-6 points per dimension.
    
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.05) # Initial small radius
    
    # Generate a hexagonal lattice and pick first 26 points
    # Lattice parameters
    dx = 0.2
    dy = 0.2 * np.sqrt(3) / 2
    
    points = []
    y = 0.5 * dy
    while y < 1.0 - 0.5 * dx: # Keep within bounds roughly
        x = 0.5 * dx
        # Offset every other row
        row_offset = 0
        if len(points) // 6 % 2 == 1: # Approximate row counting
            row_offset = dx / 2
        
        while x < 1.0 - 0.5 * dx:
            points.append([x + row_offset, y])
            x += dx
        y += dy
        
        if len(points) >= N:
            break
            
    # If we didn't get enough points (unlikely with these params), fall back to grid
    if len(points) < N:
        for i in range(N):
            row = i // 6
            col = i % 6
            centers[i] = [(col + 0.5) / 6, (row + 0.5) / 5]
    else:
        # Use the generated points
        centers[:len(points)] = points[:N]
        # If fewer than N, fill rest randomly or copy (should be enough)
        for i in range(len(points), N):
             centers[i] = centers[i % len(points)] + np.random.rand(2) * 0.01

    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    # Actually, let's optimize positions and radii separately or together.
    # Optimizing together is better.
    # Vector: [x1, y1, ..., x26, y26, r1, ..., r26]
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.extend([(0, 1), (0, 1)]) # x, y
        bounds.append((0, 0.5))          # r

    def objective(vars):
        c = vars[:2*N].reshape(N, 2)
        r = vars[2*N:]
        # We want to maximize sum(r), so minimize -sum(r)
        return -np.sum(r)

    def get_constraints_penalty(vars):
        c = vars[:2*N].reshape(N, 2)
        r = vars[2*N:]
        
        penalty = 0.0
        
        # 1. Boundary constraints
        # x >= r  => r - x <= 0
        # x <= 1-r => x + r - 1 <= 0
        # Same for y
        # We penalize violations with a squared term
        
        # Check x bounds
        # r - x < 0 -> violation amount r-x
        viol_x_low = np.maximum(0, r - c[:, 0])
        viol_x_high = np.maximum(0, c[:, 0] + r - 1)
        penalty += np.sum(viol_x_low**2) + np.sum(viol_x_high**2)
        
        # Check y bounds
        viol_y_low = np.maximum(0, r - c[:, 1])
        viol_y_high = np.maximum(0, c[:, 1] + r - 1)
        penalty += np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        # 2. Overlap constraints
        # dist(c_i, c_j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # We want to penalize (r_i + r_j)^2 - dist^2 if positive
        
        # Vectorized overlap check
        # This can be expensive for large N, but N=26 is small.
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = np.sum((c[i] - c[j])**2)
                sum_r = r[i] + r[j]
                # If dist_sq < sum_r^2, we have overlap
                # Penalty proportional to the amount of overlap squared
                # Using a smooth function: max(0, sum_r - dist)^2 is not smooth at 0?
                # Actually max(0, sum_r^2 - dist_sq)^2 is smoother?
                # Let's use: if dist < sum_r, penalty = (sum_r - dist)^2
                # But dist is sqrt. (sum_r - sqrt(dist_sq))^2.
                # This is differentiable except at dist=0.
                
                dist = np.sqrt(dist_sq + 1e-12) # Avoid div by zero
                overlap = sum_r - dist
                if overlap > 0:
                    penalty += overlap**2 * 100 # Weight for overlap
                    
        return penalty

    def penalized_objective(vars):
        base = objective(vars)
        pen = get_constraints_penalty(vars)
        # Weight for penalty needs to be high enough
        # If radii are ~0.1, sum is ~2.6. Penalty should be >> 2.6 if violated.
        return base + pen * 1000.0 

    # Run optimization
    # L-BFGS-B is good for bound constrained problems
    result = opt.minimize(penalized_objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    final_vars = result.x
    centers_opt = final_vars[:2*N].reshape(N, 2)
    radii_opt = final_vars[2*N:]
    
    # Post-processing: Clean up tiny radii or exact boundary touches if needed
    # But the penalty method should have kept them valid if weight is high.
    # Let's verify validity and potentially clamp.
    
    # Re-validate and fix any minor numerical issues
    valid_centers, valid_radii = fix_packing(centers_opt, radii_opt)
    
    sum_radii = np.sum(valid_radii)
    
    return valid_centers, valid_radii, sum_radii

def fix_packing(centers, radii):
    """
    Iteratively fixes overlaps and boundary violations by reducing radii or moving centers.
    This is a safety net for the optimizer.
    """
    N = centers.shape[0]
    centers = centers.copy()
    radii = radii.copy()
    
    # First, ensure boundaries
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        # Clamp center to be at least r away from boundary
        # If x < r, set x = r. This might cause overlap, handled later.
        # But simply clamping x=r is safe for boundary.
        if x < r: x = r
        if x > 1 - r: x = 1 - r
        if y < r: y = r
        if y > 1 - r: y = 1 - r
        centers[i] = [x, y]
        
    # Resolve overlaps
    # Simple iterative resolution: if overlap, shrink radii or move apart.
    # Since we want to maximize sum, we prefer moving apart.
    # But if stuck, shrinking is necessary.
    
    # Let's try a repulsive force approach for a few iterations to polish
    # Or just shrink overlapping circles until they touch.
    
    # Greedy shrink: Find overlapping pair, reduce radius of smaller one?
    # Or split the reduction.
    
    # Actually, a better fix is to run a local relaxation.
    # But for robustness, let's just check and shrink if strictly necessary.
    # Given the optimizer used high penalty, it should be valid.
    
    # Let's do one pass of shrinkage to guarantee validity
    changed = True
    while changed:
        changed = False
        for i in range(N):
            # Check boundaries again after moves
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9:
                radii[i] = max(0, x)
                changed = True
            elif x + r > 1 + 1e-9:
                radii[i] = max(0, 1 - x)
                changed = True
            elif y - r < -1e-9:
                radii[i] = max(0, y)
                changed = True
            elif y + r > 1 + 1e-9:
                radii[i] = max(0, 1 - y)
                changed = True
            
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                req = radii[i] + radii[j]
                if dist < req - 1e-9:
                    # Overlap
                    # Reduce radii proportionally to resolve
                    overlap = req - dist
                    # Split overlap reduction
                    delta = overlap / 2
                    radii[i] = max(0, radii[i] - delta)
                    radii[j] = max(0, radii[j] - delta)
                    changed = True
    
    return centers, radii

# Helper to generate initial grid if needed inside the function logic
# But logic is inside run_packing.

if __name__ == "__main__":
    # Self-test
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")

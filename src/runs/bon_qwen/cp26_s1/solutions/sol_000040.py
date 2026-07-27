# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 634a18b7) state=9cf69eae sum of radii=2.557337 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We attempt to fit a hexagonal grid. 
    # Approximate spacing for 26 circles:
    # Area ~ 1. Density ~ 0.9. r ~ 0.1. Spacing ~ 0.2.
    # We will use a grid and then optimize.
    
    # Create a grid of points
    # Try to fit roughly sqrt(26) ~ 5.1 in each dimension
    # Let's use a dense grid and select points, or just generate a specific pattern.
    # A 6x5 grid has 30 points. We can remove 4 or just start with 26 points 
    # distributed in a hex pattern.
    
    centers = np.zeros((n, 2))
    
    # Hexagonal packing parameters
    # Spacing dx = 2r, dy = sqrt(3)r. 
    # Let's assume r ~ 0.11 initially to spread them out.
    r_init = 0.11
    dx = 2 * r_init
    dy = math.sqrt(3) * r_init
    
    count = 0
    row = 0
    # Generate points until we have 26
    while count < n:
        y = r_init + row * dy
        if y + r_init > 1.0:
            # If row doesn't fit, we might need to shrink or shift. 
            # For initialization, just placing them and letting optimizer fix is fine.
            # But let's try to fit them.
            pass 
            
        # x positions alternate
        if row % 2 == 0:
            x_start = r_init
        else:
            x_start = r_init + dx/2
            
        col = 0
        x = x_start
        while x + r_init <= 1.0 + 1e-6: # Allow slight overflow for init
            if count < n:
                centers[count] = [x, y]
                count += 1
            x += dx
            col += 1
        row += 1
        
        # Safety break
        if row > 20:
            break

    # If we didn't fill 26 (unlikely with init logic), fill remaining randomly
    if count < n:
        for i in range(count, n):
            centers[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]

    # 2. Optimization Loop
    # We iterate: Solve LP for radii -> Compute forces -> Move centers
    
    num_iter = 1000
    step_size = 0.005
    tol_touch = 1e-6
    
    # Precompute index pairs for constraints
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            
    # LP Solver Setup Constants
    # We will reconstruct A_ub and b_ub every iteration or update distances
    
    # Current radii (initialized small)
    radii = np.full(n, 0.01)
    
    for iteration in range(num_iter):
        # A. Solve LP for optimal radii given current centers
        # Variables: r_0, ..., r_25
        # Objective: Maximize sum(r) => Minimize -sum(r)
        c_obj = np.ones(n) * -1
        
        # Constraints: A_ub * r <= b_ub
        # 1. Pairwise: r_i + r_j <= dist_ij
        # 2. Boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        
        # Number of constraints: len(pairs) + 4*n
        n_constraints = len(pairs) + 4 * n
        
        # We can use a sparse matrix or dense. n=26 is small, dense is fine.
        A_ub = np.zeros((n_constraints, n))
        b_ub = np.zeros(n_constraints)
        
        idx = 0
        for i, j in pairs:
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
        for i in range(n):
            # r_i <= x_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = centers[i, 0]
            idx += 1
            # r_i <= 1 - x_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = 1.0 - centers[i, 0]
            idx += 1
            # r_i <= y_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = centers[i, 1]
            idx += 1
            # r_i <= 1 - y_i
            A_ub[idx, i] = 1.0
            b_ub[idx] = 1.0 - centers[i, 1]
            idx += 1
            
        # Bounds for r_i: [0, 1]
        bounds = [(0, 1)] * n
        
        try:
            # Use high-performance LP solver
            res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
            else:
                # If LP fails, keep previous radii or reset
                pass
        except Exception:
            pass
            
        # B. Compute Forces for Center Movement
        # Heuristic: If circles are touching (constraint active), push them apart.
        # If circle touches boundary, push away from boundary.
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # We check if r_i + r_j is close to distance.
        # If so, the constraint is active, and separating them allows larger radii.
        # Force direction: vector from j to i.
        
        # To prevent oscillation, only apply force if "stuck"
        # Threshold for "touching"
        
        # Note: dist can be 0 if centers overlap. Handle that.
        
        for i, j in pairs:
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 1e-9:
                # Centers coincident, push randomly
                forces[i] += np.random.rand(2) - 0.5
                forces[j] -= np.random.rand(2) - 0.5
                continue
            
            unit_vec = diff / dist
            sum_r = radii[i] + radii[j]
            
            # If touching (within tolerance), apply repulsion
            # The stronger the contact, the more we want to separate?
            # Actually, if sum_r == dist, we are constrained. 
            # Moving apart increases capacity.
            # Let's apply a constant repulsive force if touching.
            if dist <= sum_r + tol_touch:
                # Repulsion force
                # Magnitude can be proportional to how "tight" the fit is?
                # Or just constant.
                force_mag = 1.0 
                forces[i] += unit_vec * force_mag
                forces[j] -= unit_vec * force_mag
        
        # Boundary repulsion
        # If r_i is close to distance to wall, push center away from wall
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall: x - r <= 0? No, constraint is r <= x.
            # If r is close to x, we are touching left wall. Push Right (+x).
            if r >= x - tol_touch:
                forces[i, 0] += 1.0
            
            # Right wall: r <= 1 - x. If r close, push Left (-x).
            if r >= (1.0 - x) - tol_touch:
                forces[i, 0] -= 1.0
                
            # Bottom wall: r <= y. If r close, push Up (+y).
            if r >= y - tol_touch:
                forces[i, 1] += 1.0
                
            # Top wall: r <= 1 - y. If r close, push Down (-y).
            if r >= (1.0 - y) - tol_touch:
                forces[i, 1] -= 1.0

        # C. Update Centers
        # Normalize forces to prevent huge jumps?
        # Or just use small step size.
        
        # Apply forces
        centers += step_size * forces
        
        # Clip centers to valid range [0, 1]
        # Actually, centers can be anywhere, but r must fit.
        # However, if center goes out of [0,1], r becomes negative (clamped by bounds).
        # But geometrically, centers should stay inside.
        # Let's clip to [0, 1] to keep valid.
        centers = np.clip(centers, 0.0, 1.0)
        
        # Damping step size to settle down?
        # Maybe keep constant or decay slowly.
        if iteration > 800:
             step_size = 0.001

    # Final LP solve to ensure radii are maximized for final centers
    c_obj = np.ones(n) * -1
    n_constraints = len(pairs) + 4 * n
    A_ub = np.zeros((n_constraints, n))
    b_ub = np.zeros(n_constraints)
    
    idx = 0
    for i, j in pairs:
        dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = dist
        idx += 1
        
    for i in range(n):
        A_ub[idx, i] = 1.0
        b_ub[idx] = centers[i, 0]
        idx += 1
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - centers[i, 0]
        idx += 1
        A_ub[idx, i] = 1.0
        b_ub[idx] = centers[i, 1]
        idx += 1
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - centers[i, 1]
        idx += 1
        
    bounds = [(0, 1)] * n
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
    except:
        pass

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper to run and validate if needed (not part of solution requirement but good for checking)
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    # Basic check
    for i in range(26):
        for j in range(i+1, 26):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-5:
                print(f"Overlap detected at {i}, {j}")
    for i in range(26):
        if centers[i][0] < radii[i] - 1e-5 or centers[i][0] > 1 - radii[i] + 1e-5:
            print(f"Boundary X violation at {i}")
        if centers[i][1] < radii[i] - 1e-5 or centers[i][1] > 1 - radii[i] + 1e-5:
            print(f"Boundary Y violation at {i}")

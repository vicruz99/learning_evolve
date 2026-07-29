# sol_000080 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ed1177e6) state=46a49f47 sum of radii=2.082311 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import random

def get_wall_distances(centers):
    """Calculate distance from each center to the nearest wall."""
    n = centers.shape[0]
    w = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        w[i] = min(x, 1 - x, y, 1 - y)
    return w

def solve_radii_lp(centers):
    """
    Given centers, solve LP to find maximum radii.
    Maximize sum(r_i)
    Subject to:
      r_i + r_j <= dist(c_i, c_j) for all i < j
      r_i <= dist(c_i, wall) for all i
      r_i >= 0
    """
    n = centers.shape[0]
    
    # Variables: r_0, ..., r_{n-1}
    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Constraints are of form r_i + r_j <= d_ij and r_i <= w_i
    
    A_ub = []
    b_ub = []
    
    # Pairwise distance constraints
    # Number of pairs: n*(n-1)/2
    # We can construct A_ub row by row or use sparse, but for n=26 dense is fine.
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    # Wall distance constraints
    walls = get_wall_distances(centers)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(walls[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Using high-performance method
    res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return -res.fun, res.x
    else:
        # Fallback if LP fails (should not happen if feasible)
        return 0, np.zeros(n)

def compute_repulsive_forces(centers, radii):
    """
    Compute forces to push centers apart to allow larger radii.
    Force is applied if circles are 'touching' or close to touching.
    Actually, a simple repulsive force based on overlap or proximity works well.
    Here, we define a target distance for each pair based on current radii?
    No, we want to maximize radii.
    If r_i + r_j < dist, there is slack.
    If r_i + r_j approx dist, they are constrained.
    
    A common heuristic:
    Force on i from j: F_ij = k * ( (r_i + r_j) / dist - 1 ) * direction ?
    If dist > r_i + r_j, force is negative (attractive)? No, we want repulsion.
    Actually, if dist > r_i + r_j, we have room to grow.
    If dist < r_i + r_j (impossible in valid config), we must separate.
    
    Better approach for optimization:
    We want to move centers to regions where constraints are loose.
    The gradient of the objective w.r.t position is complex.
    Heuristic: Repulsion force inversely proportional to distance?
    Or repulsion only when close?
    
    Let's use a standard repulsive force: F = 1/d^2.
    This tends to spread points out.
    Also repulsion from walls.
    """
    n = centers.shape[0]
    forces = np.zeros((n, 2))
    
    # Pairwise repulsion
    # Using a soft repulsion to avoid singularity at 0
    # F = (r_i + r_j) / max(dist, small_epsilon)^2 ?
    # If we use just 1/d^2, it spreads them uniformly.
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[j] - centers[i]
            dist_sq = np.dot(diff, diff)
            dist = np.sqrt(dist_sq)
            if dist < 1e-6:
                # Random push if coincident
                force_vec = np.random.rand(2) * 0.01
            else:
                # Repulsive force magnitude
                # We want to push them apart.
                # Strength can depend on radii?
                # If radii are large, they need more space.
                # Let's use F ~ (r_i + r_j) / dist^2
                mag = (radii[i] + radii[j]) / (dist_sq + 1e-6)
                force_vec = (diff / dist) * mag
            
            forces[i] -= force_vec
            forces[j] += force_vec

    # Wall repulsion
    # If close to wall, push away
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left wall
        if x < r + 0.05: # Proximity threshold
            forces[i, 0] += (r + 0.05 - x) * 10.0
        # Right wall
        if x > 1 - r - 0.05:
            forces[i, 0] -= (x - (1 - r - 0.05)) * 10.0
        # Bottom
        if y < r + 0.05:
            forces[i, 1] += (r + 0.05 - y) * 10.0
        # Top
        if y > 1 - r - 0.05:
            forces[i, 1] -= (y - (1 - r - 0.05)) * 10.0
            
    return forces

def update_centers(centers, forces, step_size):
    """Update centers based on forces, keeping within bounds."""
    new_centers = centers + step_size * forces
    
    # Clamp to [0, 1]
    # Note: Centers should ideally stay away from boundaries by at least some margin,
    # but clamping to [0,1] is safe. The radius constraint handles the rest.
    # Actually, if center is at 0, radius must be 0.
    # So we should clamp to [epsilon, 1-epsilon]?
    # But let's just clamp to [0, 1] and let LP handle radius=0 if needed.
    # However, to prevent getting stuck at 0, maybe clamp to [1e-4, 1-1e-4].
    
    new_centers = np.clip(new_centers, 1e-4, 1 - 1e-4)
    return new_centers

def run_packing():
    n = 26
    # Initial placement: Hexagonal lattice
    # We need to fit 26 circles.
    # Approx radius 0.1.
    # Rows:
    # Row 1: 5 circles
    # Row 2: 5 circles
    # ...
    # Let's try to pack them densely.
    
    centers = np.zeros((n, 2))
    
    # Simple grid init first, then we optimize
    # 5x5 grid is 25. Add 1 in center?
    # Or just random init in a box?
    # Let's try a perturbed grid.
    
    # 5 rows, 6 cols? No.
    # Let's place them in a grid 6x5 = 30 spots, pick 26?
    # Or just specific coordinates.
    
    # Hexagonal packing init
    # Row height: sqrt(3)/2 * diameter ~ 0.866 * 2r.
    # If r=0.1, diam=0.2, row height=0.1732.
    # 5 rows height ~ 0.2 + 4*0.1732 = 0.89. Fits.
    # Row width: 5 circles -> width 0.8 + margins.
    
    r_init = 0.09
    dx = 2 * r_init * 1.1 # slight spacing
    dy = r_init * np.sqrt(3) * 1.1
    
    row = 0
    col = 0
    count = 0
    
    # Try to fill rows
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    
    # Let's just place 26 circles in a reasonable grid
    # 5x5 grid centers at 0.2, 0.4, 0.6, 0.8, 1.0? No, 0.1 to 0.9.
    # x in [0.2, 0.4, 0.6, 0.8] -> 4 circles?
    # If r=0.1, centers at 0.1, 0.3, 0.5, 0.7, 0.9 -> 5 circles.
    
    # Let's create a 5x6 grid and pick 26 best spots?
    # Or just 5x5 + 1.
    
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    
    # 25 circles in 5x5
    idx = 0
    for y in ys:
        for x in xs:
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
            else:
                break
        if idx >= n:
            break
            
    # If we have 25, place 26th in a gap?
    # Gaps are at (0.2, 0.2), (0.2, 0.4)... relative to grid?
    # Centers at 0.1, 0.3...
    # Gaps at midpoints: 0.2, 0.2.
    if n > 25:
        # Place 26th circle at (0.5, 0.5) - center of square?
        # Or a gap.
        # Let's place at (0.5, 0.5) which is center of a 4-circle hole in 5x5?
        # 5x5 holes are at (0.2, 0.2), (0.4, 0.2)...
        # (0.5, 0.5) is a center of a circle in 5x5?
        # 0.1, 0.3, 0.5, 0.7, 0.9. Yes, 0.5 is a center.
        # So (0.5, 0.5) is occupied.
        # Let's pick a gap: (0.2, 0.2).
        centers[25] = [0.2, 0.2]
        
    # Better init: Hexagonal
    # Let's regenerate centers in hex pattern
    centers = np.zeros((n, 2))
    count = 0
    y = 0.15 # start y
    while count < n:
        # Determine number of circles in this row
        # If row index (starting 0) is even, 5 circles?
        # If odd, 4 circles (shifted)?
        # Let's try to fit as many as possible.
        # Width 1.0. Diameter ~ 0.2. Max 5.
        
        # Shift
        row_idx = count // 5 # Approximate
        is_shifted = (row_idx % 2 == 1)
        
        if is_shifted:
            x_start = 0.1 + 0.1 # Shift by r
            num_in_row = 4
        else:
            x_start = 0.1
            num_in_row = 5
            
        x = x_start
        while count < n and num_in_row > 0:
            centers[count] = [x, y]
            x += 0.2
            count += 1
            num_in_row -= 1
            
        y += 0.17 # Vertical spacing
        
    # Optimization Loop
    radii = np.zeros(n)
    best_sum = 0
    best_centers = centers.copy()
    best_radii = radii
    
    # Parameters
    step_size = 0.001
    max_iters = 500
    
    for it in range(max_iters):
        # 1. Solve LP for radii
        current_sum, radii = solve_radii_lp(centers)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # 2. Compute forces
        forces = compute_repulsive_forces(centers, radii)
        
        # 3. Update centers
        # Adaptive step size?
        # If improvement, keep step size. Else reduce?
        # For simplicity, just update.
        centers = update_centers(centers, forces, step_size)
        
        # Decay step size slowly
        step_size *= 0.999
        
        # Stop if radii are very small or stuck?
        if np.all(radii < 1e-5):
            break

    # Return best found
    return best_centers, best_radii, best_sum

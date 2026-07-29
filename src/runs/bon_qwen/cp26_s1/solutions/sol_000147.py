# sol_000147 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e234a3e4) state=de83068f sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed simulation for center optimization followed by 
    Linear Programming to determine optimal radii.
    """
    n_circles = 26
    np.random.seed(42)

    # --- Stage 1: Initialization ---
    # Generate a hexagonal grid pattern to start with a dense configuration.
    # Approximate spacing for 26 circles in a square.
    # sqrt(26) ~ 5.1. Let's try fitting roughly 6x5 or similar.
    # Hexagonal lattice:
    # Row height = sqrt(3)/2 * spacing
    # Col width = spacing
    
    # We want to cover the [0,1]x[0,1] square.
    # Let's try to fit 6 columns and 5 rows? 30 points. We need 26.
    # We can generate a larger grid and pick the first 26, or distribute them.
    
    cols = 6
    rows = 5
    spacing_x = 1.0 / (cols - 0.5) # Approximate
    spacing_y = 1.0 / (rows - 0.5)
    
    # Better hexagonal initialization logic:
    # We want to tile the square.
    # Let's create a list of potential centers and pick n_circles best ones?
    # Or just place them in a staggered grid.
    
    centers = []
    # Try to fit circles in rows.
    # Row 0: centers at x = 0.5/cols, 3*0.5/cols...
    # Actually, let's just generate a dense hex grid and take 26 points.
    
    # Estimate optimal radius for 26 circles ~ 0.1. Diameter 0.2.
    # So spacing ~ 0.2.
    # 1/0.2 = 5. So 5x5 grid is 25.
    # We need 26.
    
    # Let's create a grid of points
    grid_points = []
    # Hexagonal lattice generation
    # Horizontal shift for odd rows
    h_shift = 0.5 * 0.2 # approximate shift
    v_shift = np.sqrt(3)/2 * 0.2
    
    # Let's just place 26 points roughly uniformly using a grid approach
    # 5 rows. 6, 5, 6, 5, 4 ? Sum = 26.
    # Row y-coords: 0.1, 0.3, 0.5, 0.7, 0.9 (approx)
    
    y_coords = np.linspace(0.1, 0.9, 5)
    
    current_y = 0
    point_idx = 0
    # Pattern: 6, 5, 6, 5, 4 is 26? 6+5+6+5+4 = 26.
    # Or 5, 6, 5, 6, 4?
    # Let's try to distribute evenly.
    
    row_counts = [6, 5, 6, 5, 4] # Sum 26
    
    for r_idx, count in enumerate(row_counts):
        y = (r_idx + 0.5) / 5.0 # Center of rows roughly
        # Adjust y for hex packing?
        # Let's just use regular y for now, optimizer will fix it.
        # Actually, let's stagger x for odd rows (indices 1, 3)
        
        # If count is even, maybe centered?
        # Let's just space them linearly in [0,1]
        # To fit 'count' circles, spacing is 1/(count+1) ?
        # Centers at (k+0.5)/count ? No, boundaries matter.
        
        # Let's place centers at:
        # x_k = (k + 0.5) / count ?
        # Range [0, 1].
        
        # Staggering:
        shift = 0
        if r_idx % 2 == 1:
            shift = 1.0 / (2 * count) # Half step shift
        
        # Generate x coordinates
        # If we have 'count' circles, maybe spacing 1/(count)?
        # Let's try to fit them with some margin.
        
        # A simple way: linspace with margin
        margin = 0.05
        x_vals = np.linspace(margin, 1-margin, count)
        
        # Apply shift for hexagonal look
        # But x_vals are already spaced. Shift might push out of bounds.
        # Let's just use the x_vals as is, but shift by a fraction of spacing if needed.
        # Or just don't shift x, rely on optimizer.
        
        for x in x_vals:
            if point_idx < n_circles:
                centers.append([x, y])
                point_idx += 1
            else:
                break
    
    # If we didn't get 26 (due to logic), fill remaining randomly
    while len(centers) < n_circles:
        centers.append([np.random.rand(), np.random.rand()])
    
    centers = np.array(centers[:n_circles])
    
    # Add small random noise to break symmetry
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, 0.0, 1.0)

    # --- Stage 2: Optimization ---
    # Force-directed layout to maximize space between circles
    
    n_steps = 1000
    lr = 0.05 # Learning rate (step size)
    radii_est = np.full(n_circles, 0.05) # Initial estimate for repulsion strength
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # 1. Repulsion from neighbors
        # We treat all circles as having radius radii_est[i]
        # Force is proportional to overlap or inverse distance
        # To maximize sum of radii, we want to maximize distances.
        # Simple repulsive force: F = 1 / dist^2 (or similar)
        
        # Vectorized distance calculation
        # diff = centers[i] - centers[j]
        # dist = norm(diff)
        # if dist < threshold, push apart
        
        # Efficient neighbor search is hard without spatial index, but N=26 is small.
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq)
                
                if dist < 1e-6:
                    dist = 1e-6
                    diff = np.random.rand(2) * 1e-6 # Avoid division by zero
                
                # Desired separation
                # If we want to maximize radii, we effectively want to maximize min distance.
                # Force magnitude
                repulsion = 0.5 / dist # Soft repulsion
                
                direction = diff / dist
                
                # Apply force
                force_vec = repulsion * direction
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # 2. Boundary repulsion
        # Push away from walls
        # Force = k * distance_to_wall? No, push IN if outside, but we clip.
        # Actually, we want to stay away from walls to allow radius.
        # So if close to wall, push inward.
        
        boundary_force_strength = 1.0
        for i in range(n_circles):
            x, y = centers[i]
            
            # Left wall
            if x < 0.1:
                forces[i, 0] += boundary_force_strength * (0.1 - x)
            # Right wall
            elif x > 0.9:
                forces[i, 0] -= boundary_force_strength * (x - 0.9)
            
            # Bottom wall
            if y < 0.1:
                forces[i, 1] += boundary_force_strength * (0.1 - y)
            # Top wall
            elif y > 0.9:
                forces[i, 1] -= boundary_force_strength * (y - 0.9)

        # Update centers
        # Decay learning rate
        current_lr = lr * (1.0 - step / n_steps)
        centers += current_lr * forces
        
        # Clip to valid range
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        
        # Update estimated radii based on current distances to neighbors
        # This helps the repulsion force adapt as circles move closer/further
        new_radii_est = np.full(n_circles, 0.0)
        for i in range(n_circles):
            min_dist = 1.0
            # Distance to boundaries
            d_bound = min(centers[i, 0], 1.0 - centers[i, 0], 
                          centers[i, 1], 1.0 - centers[i, 1])
            min_dist = min(min_dist, d_bound)
            
            # Distance to other centers
            dists = np.sqrt(np.sum((centers - centers[i])**2, axis=1))
            dists[i] = np.inf # Ignore self
            min_dist_neighbor = np.min(dists)
            # Radius is roughly half the distance to nearest obstacle
            # But for force scaling, let's use a fraction
            new_radii_est[i] = min_dist * 0.5
            
        radii_est = new_radii_est

    # --- Stage 3: Solve for optimal radii using LP ---
    # Maximize sum(r_i)
    # Subject to:
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # r_i + r_j <= dist(c_i, c_j)
    # r_i >= 0
    
    # Variables: r_0, ..., r_25
    # Objective: -sum(r_i) (for minimization)
    c_obj = np.ones(n_circles)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    # r_i <= x_i  => 1*r_i <= x_i
    # r_i <= 1-x_i => 1*r_i <= 1-x_i
    # r_i <= y_i
    # r_i <= 1-y_i
    
    for i in range(n_circles):
        # x constraints
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(centers[i, 0])
        
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - centers[i, 0])
        
        # y constraints
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(centers[i, 1])
        
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - centers[i, 1])

    # Overlap constraints
    # r_i + r_j <= dist_ij
    # 1*r_i + 1*r_j <= dist_ij
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n_circles)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for r_i: [0, None]
    bounds = [(0, None) for _ in range(n_circles)]
    
    # Solve LP
    # Using 'highs' method if available, else default
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        optimal_radii = res.x
        sum_radii = np.sum(optimal_radii)
    else:
        # Fallback if LP fails (should not happen with valid centers)
        optimal_radii = np.zeros(n_circles)
        sum_radii = 0.0
        
    return centers, optimal_radii, sum_radii

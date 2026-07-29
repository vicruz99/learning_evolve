# sol_000022 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a097d99c) state=c4b7550b sum of radii=1.382756 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_pair_indices(n):
    """Generate indices for all unique pairs (i, j) with i < j."""
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            indices.append((i, j))
    return indices

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 1. Initialization: Grid layout
    # 5x5 grid gives 25 circles. We add one more in a gap or just perturb.
    # A 6x5 grid would be 30, so let's pick 26 points from a grid.
    # Let's use a 6x5 grid of points, but we only need 26.
    # Actually, 5 rows of 5 is 25. Let's do 5 rows of 5, and place the 26th 
    # in a corner or gap.
    
    # Better initialization: Hexagonal packing start or just dense grid.
    # Let's try to fit them in a 6x5 pattern roughly.
    # 6 columns, 5 rows = 30 slots. We pick 26.
    
    # Let's just use a simple 6x6 grid of potential spots and pick first 26
    # Spacing 0.2 is tight.
    
    # Strategy: Start with valid small circles.
    # We can place them in a spiral or just a dense grid.
    
    # Let's use a 6x5 grid of centers with some padding.
    # x = 0.1, 0.3, 0.5, 0.7, 0.9, 1.1 (invalid)
    # Let's do 5 columns, 6 rows.
    # x in [0.1, 0.3, 0.5, 0.7, 0.9] (5 cols)
    # y in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1] (invalid)
    
    # Let's use a slightly randomized grid to break symmetry and avoid local optima.
    np.random.seed(42)
    
    # Generate 26 points
    # Try to distribute them somewhat evenly
    points = []
    # 5 rows, 5 cols = 25. Add 1.
    for i in range(5):
        for j in range(5):
            points.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    # Add 26th point in a gap, e.g., center of square if empty, or corner
    # Center (0.5, 0.5) is occupied.
    # Let's put it at (0.05, 0.05) or similar.
    points.append([0.05, 0.05])
    
    centers = np.array(points)
    # Initial radii
    radii = np.full(n, 0.001)

    pair_indices = get_pair_indices(n)
    
    # Optimization Loop
    num_iterations = 2000
    step_size = 0.005
    
    for it in range(num_iterations):
        # Phase 1: Solve LP for optimal radii given current centers
        # Maximize sum(r) -> Minimize -sum(r)
        # Variables: r_0, ..., r_25
        
        # Objective coefficients
        c_obj = -np.ones(n)
        
        # Constraints: A_ub @ r <= b_ub
        # 1. Pairwise constraints: r_i + r_j <= dist(i, j)
        # 2. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        
        n_constraints = len(pair_indices) + 4 * n
        A_ub = np.zeros((n_constraints, n))
        b_ub = np.zeros(n_constraints)
        
        row = 0
        for (i, j) in pair_indices:
            dist = np.linalg.norm(centers[i] - centers[j])
            A_ub[row, i] = 1.0
            A_ub[row, j] = 1.0
            b_ub[row] = dist
            row += 1
            
        for i in range(n):
            # x bounds
            A_ub[row, i] = 1.0
            b_ub[row] = centers[i, 0]
            row += 1
            # 1-x bounds
            A_ub[row, i] = 1.0
            b_ub[row] = 1.0 - centers[i, 0]
            row += 1
            # y bounds
            A_ub[row, i] = 1.0
            b_ub[row] = centers[i, 1]
            row += 1
            # 1-y bounds
            A_ub[row, i] = 1.0
            b_ub[row] = 1.0 - centers[i, 1]
            row += 1
            
        # Bounds for r: r >= 0
        bounds = [(0, None) for _ in range(n)]
        
        # Solve LP
        # Use high precision method if possible, but default is usually fine
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # If LP fails, keep previous radii or break?
            # Break to avoid instability
            break
            
        radii = res.x
        current_sum = np.sum(radii)
        
        # Phase 2: Calculate forces and move centers
        forces = np.zeros_like(centers)
        
        # Repulsive forces between touching circles
        # We want to push apart circles that are "tight" (slack close to 0)
        # Slack s = dist - (r_i + r_j)
        
        for (i, j) in pair_indices:
            r_i = radii[i]
            r_j = radii[j]
            dist = np.linalg.norm(centers[i] - centers[j])
            required_dist = r_i + r_j
            
            # Calculate slack
            slack = dist - required_dist
            
            # If slack is small (negative means overlap, but LP ensures >= 0, floating point might be tiny negative)
            # We treat small slack as tight constraint.
            # We want to push them apart if slack is small.
            # Force magnitude proportional to 1/(slack + epsilon) ?
            # Or just a constant repulsion if they are close?
            # Let's use a force that decays with distance but is strong when tight.
            # Actually, simpler: if slack < threshold, push apart.
            
            if slack < 0.005: # Threshold for "touching"
                # Vector from j to i
                if dist > 1e-9:
                    dir_vec = (centers[i] - centers[j]) / dist
                else:
                    dir_vec = np.array([0.0, 0.0]) # Random push if same pos?
                
                # Force strength: stronger when tighter
                # Use inverse of slack (clamped)
                strength = 0.01 / (slack + 0.001)
                
                forces[i] += dir_vec * strength
                forces[j] -= dir_vec * strength
        
        # Boundary forces
        # If circle is touching wall, push inside
        for i in range(n):
            r_i = radii[i]
            # Left wall
            slack_l = centers[i, 0] - r_i
            if slack_l < 0.005:
                forces[i, 0] += 0.1 / (slack_l + 0.001)
            
            # Right wall
            slack_r = (1.0 - centers[i, 0]) - r_i
            if slack_r < 0.005:
                forces[i, 0] -= 0.1 / (slack_r + 0.001)
            
            # Bottom wall
            slack_b = centers[i, 1] - r_i
            if slack_b < 0.005:
                forces[i, 1] += 0.1 / (slack_b + 0.001)
            
            # Top wall
            slack_t = (1.0 - centers[i, 1]) - r_i
            if slack_t < 0.005:
                forces[i, 1] -= 0.1 / (slack_t + 0.001)
        
        # Update centers
        # Apply forces with a step size
        # Adaptive step size?
        # Let's just use fixed small step
        centers = centers + step_size * forces
        
        # Clamp centers to [0, 1]
        centers = np.clip(centers, 0.0, 1.0)

    # Final validation and return
    # Ensure radii are consistent with final centers (solve LP one last time)
    # This is important because we moved centers, radii might be invalid or suboptimal
    c_obj = -np.ones(n)
    n_constraints = len(pair_indices) + 4 * n
    A_ub = np.zeros((n_constraints, n))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    for (i, j) in pair_indices:
        dist = np.linalg.norm(centers[i] - centers[j])
        A_ub[row, i] = 1.0
        A_ub[row, j] = 1.0
        b_ub[row] = dist
        row += 1
        
    for i in range(n):
        A_ub[row, i] = 1.0
        b_ub[row] = centers[i, 0]
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = 1.0 - centers[i, 0]
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = centers[i, 1]
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = 1.0 - centers[i, 1]
        row += 1
        
    bounds = [(0, None) for _ in range(n)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
    
    sum_radii = np.sum(radii)
    
    # Debug check
    # validate_packing(centers, radii)
    
    return centers, radii, sum_radii

# To verify the function works locally, we can define the validation function
# but the prompt says we don't modify it and it's provided.
# I will just output the run_packing function.

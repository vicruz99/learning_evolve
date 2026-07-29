# sol_000098 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 15bab5cf) state=c4b0a0cb sum of radii=1.227020 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Grid
    # We aim to distribute points roughly uniformly.
    # A hexagonal lattice is a good starting point.
    # We will generate a grid of points and select/arrange them to fit n_circles.
    
    # Estimate spacing. Area per circle ~ 1/26. 
    # Side of square area ~ sqrt(1/26) ~ 0.196.
    # Spacing between centers ~ 2 * r ~ 0.2 - 0.25.
    
    # Let's try to fit points in a hexagonal arrangement.
    # Rows y_k. Columns x_j.
    # y_k = k * dy. x_j = j * dx + (k % 2) * dx / 2.
    # We want to fit roughly 5 rows.
    
    points = []
    
    # Let's construct a 5-row hexagonal grid.
    # Row 0: 5 points
    # Row 1: 5 points (shifted)
    # Row 2: 5 points
    # Row 3: 5 points
    # Row 4: 6 points? Or adjust counts to sum to 26.
    # 5+5+5+5+6 = 26.
    # Let's try to fit this.
    
    # We need to determine dx, dy.
    # In hex packing, dist = 2r. dy = sqrt(3)/2 * 2r = sqrt(3) r. dx = 2r.
    # Ratio dy/dx = sqrt(3)/2 approx 0.866.
    
    # Let's assume a target radius r ~ 0.1.
    # dx = 0.2, dy = 0.1732.
    # Total height for 5 rows (4 gaps) = 4 * dy = 0.69.
    # Plus margins r=0.1 at top/bottom -> 0.89. Fits in 1.
    # Width for 6 points (5 gaps) = 5 * dx = 1.0.
    # Plus margins -> 1.2. Too wide.
    # So 6 points in a row is tight.
    # Maybe 6, 5, 5, 5, 5 is better distributed?
    # Or 5, 5, 5, 6, 5?
    
    # Let's try a simpler approach: Distribute on a 5x5 grid (25 pts) + 1 pt.
    # Grid points at 0.1, 0.3, 0.5, 0.7, 0.9.
    # Add 26th point at (0.5, 0.5)? It's occupied.
    # Add at (0.25, 0.25)?
    
    # Better: Generate a dense hex grid and pick 26 points that fit well.
    # Or just random initialization and let optimizer work?
    # Random is risky.
    
    # Let's generate a 6x5 rectangular grid (30 points) and pick 26?
    # No, just place 26 points in a pattern.
    # Let's try to place 26 points in a 5x5 grid pattern but perturbed?
    # Actually, a 5x5 grid has 25 points.
    # We can place 26th point in the center of the square (0.5, 0.5) is occupied by a grid point?
    # Grid: 0.1, 0.3, 0.5, 0.7, 0.9.
    # (0.5, 0.5) is a center.
    # If we remove one and add two? No.
    
    # Let's use a "spiral" or just a structured grid that sums to 26.
    # Rows: 6, 5, 5, 5, 5.
    # Row 0 (6 pts): y = 0.1 + 0.0? No.
    # Let's define y coordinates for 5 rows evenly spaced.
    # y_vals = np.linspace(0.1, 0.9, 5) -> [0.1, 0.3, 0.5, 0.7, 0.9]
    # Row 0: 6 points. x = linspace(0.1, 0.9, 6)?
    # linspace(0.1, 0.9, 6) -> 0.1, 0.26, 0.42, 0.58, 0.74, 0.9.
    # Spacing ~ 0.16. 2r would be 0.16 -> r=0.08.
    # Row 1: 5 points. x = linspace(0.1, 0.9, 5) -> 0.1, 0.3, 0.5, 0.7, 0.9.
    # Spacing 0.2.
    # This creates a valid initial guess, though radii will be limited by the dense row.
    # Optimizer will spread them out.
    
    initial_centers = []
    
    # Configuration: 6, 5, 5, 5, 5
    row_counts = [6, 5, 5, 5, 5]
    num_rows = len(row_counts)
    
    # Y coordinates for rows
    # We want to utilize vertical space.
    # If r ~ 0.1, we need margins.
    # Let's place rows at y = 0.1, 0.3, 0.5, 0.7, 0.9 initially.
    row_y = np.linspace(0.1, 0.9, num_rows)
    
    for r_idx, count in enumerate(row_counts):
        y = row_y[r_idx]
        # X coordinates
        # To allow optimizer to adjust, space them somewhat evenly in [0.1, 0.9]
        # Or [0, 1] but constrained later. Let's put in [0.05, 0.95]
        x_vals = np.linspace(0.05, 0.95, count)
        for x in x_vals:
            initial_centers.append([x, y])
            
    # If we have more/less than 26, adjust.
    # 6+5+5+5+5 = 26. Correct.
    
    initial_centers = np.array(initial_centers)
    
    # 2. Optimization
    # Minimize repulsive potential energy to spread points apart.
    # E = sum(1 / dist^2)
    # Bounds [0, 1] for all coords.
    
    def energy_func(pos):
        # pos is flattened array of size 2 * n_circles
        centers = pos.reshape(-1, 2)
        total_energy = 0.0
        # Compute pairwise distances
        # Using broadcasting for efficiency
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # dist_sq = dist_sq[np.triu_indices_from(dist_sq, k=1)] # Upper triangle
        
        # Loop is fine for 26 points (325 pairs)
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d_sq = dx*dx + dy*dy
                # Avoid division by zero, add small epsilon
                # Also cap minimum distance to avoid huge gradients
                # But optimization should keep them apart.
                d_sq = max(d_sq, 1e-6) 
                total_energy += 1.0 / d_sq
        return total_energy

    # Initial position array
    x0 = initial_centers.flatten()
    
    # Bounds
    bounds = [(0, 1)] * (2 * n_circles)
    
    # Run optimization
    # L-BFGS-B is good for bound constrained problems
    result = minimize(energy_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
    
    optimal_centers = result.x.reshape(-1, 2)
    
    # 3. Calculate Radii
    # For each circle, radius is limited by distance to boundaries and other circles.
    # r_i = min( dist_to_boundary, 0.5 * min_dist_to_other_centers )
    
    radii = np.zeros(n_circles)
    
    # Precompute distances
    # dist_matrix[i, j] = distance between i and j
    dist_matrix = np.zeros((n_circles, n_circles))
    for i in range(n_circles):
        for j in range(n_circles):
            if i == j:
                dist_matrix[i, j] = np.inf
            else:
                d = np.sqrt(np.sum((optimal_centers[i] - optimal_centers[j])**2))
                dist_matrix[i, j] = d
                
    for i in range(n_circles):
        x, y = optimal_centers[i]
        # Distance to boundaries
        dist_boundary = min(x, 1-x, y, 1-y)
        
        # Distance to nearest neighbor
        min_dist_neighbor = np.min(dist_matrix[i, :])
        
        # Radius constrained by boundary and neighbors
        # Neighbors constraint: r_i + r_j <= d_ij
        # If we assume equal radii locally or just compute max possible r_i assuming others are 0?
        # No, we need a consistent set.
        # However, a safe lower bound for r_i given current positions is:
        # r_i <= dist_boundary
        # r_i <= (d_ij - r_j) for all j.
        # This is a system of inequalities.
        # But a simple heuristic for "sum of radii" maximization given fixed centers
        # is to just set r_i = min(dist_boundary, 0.5 * min_dist_neighbor).
        # This assumes neighbors have same radius? 
        # Actually, if r_i = 0.5 * d_ij, then r_i + r_j = d_ij implies r_i = r_j.
        # So this sets all touching circles to equal radius.
        # Since the optimizer tries to equalize distances, this should result in roughly equal radii.
        
        r_i = min(dist_boundary, 0.5 * min_dist_neighbor)
        radii[i] = r_i
        
    # 4. Refinement / Validation
    # The simple radius calculation might result in slight overlaps if not careful, 
    # but mathematically r_i <= 0.5 * d_ij ensures r_i + r_j <= d_ij IF r_j is also <= 0.5 * d_ij.
    # Since d_ij = d_ji, 0.5 * d_ij is symmetric.
    # So r_i + r_j <= 0.5*d_ij + 0.5*d_ij = d_ij.
    # So no overlap.
    # Also r_i <= dist_boundary ensures inside square.
    
    # However, we might be able to increase radii sum if some circles are in "gaps".
    # But with optimal packing, gaps are small.
    
    # Let's verify and maybe slightly scale up if there is global slack?
    # If min(r_i) is very small, maybe we can scale up?
    # But radii are limited by local constraints.
    
    # One check: The optimizer maximized min-distance.
    # The minimum distance in the set is min_pairwise_dist.
    # The minimum boundary distance is min_boundary_dist.
    # The bottleneck radius R_bottleneck = min(min_pairwise_dist / 2, min_boundary_dist).
    # If we set all radii to R_bottleneck, sum is 26 * R_bottleneck.
    # But variable radii allows some to be larger.
    # Our calculation r_i = min(dist_boundary, 0.5 * min_neighbor_dist) captures this.
    # Note: min_neighbor_dist for circle i is distance to closest circle j.
    # Let d_min_i be that distance.
    # r_i = min(boundary_i, d_min_i / 2).
    # Is this consistent?
    # Suppose circle i and j are neighbors. d_ij is the distance.
    # d_min_i <= d_ij. d_min_j <= d_ij.
    # r_i <= d_min_i / 2 <= d_ij / 2.
    # r_j <= d_min_j / 2 <= d_ij / 2.
    # r_i + r_j <= d_ij.
    # So valid.
    
    # Is it maximal?
    # For a specific circle i, r_i is limited by boundary and all neighbors.
    # r_i <= boundary_i.
    # r_i <= d_ij - r_j.
    # This is a dependency.
    # However, if the configuration is "tight" (circles touching), the simple assignment works well.
    # If there are gaps, r_i will be larger.
    
    # To be safe and potentially improve sum, we could solve the system:
    # maximize sum(r) s.t. r_i + r_j <= d_ij, r_i <= boundary_i.
    # This is a linear programming problem!
    # Variables r_1...r_26.
    # Maximize sum(r_i).
    # Constraints:
    # r_i + r_j <= dist(i, j) for all i < j
    # r_i <= boundary_dist(i) for all i
    # r_i >= 0
    
    # This is much better than the heuristic. It will find the optimal radii for the fixed centers.
    
    # Let's implement this LP.
    from scipy.optimize import linprog
    
    # Number of variables = 26
    # Objective: minimize -sum(r_i)
    c = -np.ones(n_circles)
    
    # Inequality constraints A_ub * r <= b_ub
    # r_i + r_j <= d_ij  =>  r_i + r_j <= d_ij
    # r_i <= b_i => r_i <= b_i
    
    # Construct A_ub
    # Rows for pairwise constraints: n_circles * (n_circles - 1) / 2
    # Rows for boundary constraints: n_circles
    
    n_pairs = n_circles * (n_circles - 1) // 2
    A_ub = np.zeros((n_pairs + n_circles, n_circles))
    b_ub = np.zeros(n_pairs + n_circles)
    
    row_idx = 0
    # Pairwise
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            A_ub[row_idx, i] = 1.0
            A_ub[row_idx, j] = 1.0
            b_ub[row_idx] = dist_matrix[i, j]
            row_idx += 1
            
    # Boundary
    for i in range(n_circles):
        x, y = optimal_centers[i]
        b_dist = min(x, 1-x, y, 1-y)
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = b_dist
        row_idx += 1
        
    # Bounds for r_i: r_i >= 0
    r_bounds = [(0, None)] * n_circles
    
    # Solve LP
    lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=r_bounds, method='highs')
    
    if lp_result.success:
        radii = lp_result.x
    else:
        # Fallback to heuristic if LP fails
        for i in range(n_circles):
            x, y = optimal_centers[i]
            b_dist = min(x, 1-x, y, 1-y)
            min_d = np.min(dist_matrix[i, :])
            radii[i] = min(b_dist, 0.5 * min_d)

    sum_radii = np.sum(radii)
    
    return optimal_centers, radii, sum_radii

# Allow running the function
if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    print(f"Centers:\n{centers}")
    print(f"Radii:\n{radii}")

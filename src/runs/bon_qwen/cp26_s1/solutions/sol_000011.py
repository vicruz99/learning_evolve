# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 60d0e48a) state=8026114e sum of radii=2.583582 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def solve_lp_for_radii(centers):
    """
    Given fixed centers, solve the LP to maximize sum of radii.
    Returns: optimal radii, sum of radii, dual variables (marginals) for constraints.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Variables: r_0, ..., r_{n-1}
    # Minimize: -sum(r_i)  => c = -1 for all
    c_obj = -np.ones(n)
    
    # Constraints: A_ub * r <= b_ub
    # We need to construct A_ub and b_ub
    
    # 1. Boundary constraints:
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # Total 4*n constraints
    
    # 2. Distance constraints:
    # r_i + r_j <= dist(i, j)
    # Total n*(n-1)/2 constraints
    
    num_ineq = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    
    # Boundary constraints
    for i in range(n):
        # r_i <= x_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = x[i]
        idx += 1
        
        # r_i <= 1 - x_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - x[i]
        idx += 1
        
        # r_i <= y_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = y[i]
        idx += 1
        
        # r_i <= 1 - y_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - y[i]
        idx += 1
        
    # Distance constraints
    # r_i + r_j <= dist
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    # Bounds for r_i: [0, inf)
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Using 'highs' method which is robust and returns duals in newer scipy
    # Fallback to 'interior-point' if needed, but highs is standard now.
    try:
        res = scipy.optimize.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except:
        res = scipy.optimize.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
    
    if res.success:
        radii = res.x
        # marginals for inequality constraints are stored in res.ineqlin.marginals
        # Note: For minimization, marginals are typically non-negative (shadow price of increasing RHS)
        # However, scipy documentation says for min c^Tx s.t. Ax<=b, marginals are -dual.
        # Let's check sign. Usually, if constraint is active and we relax it (increase b), obj improves (decreases for min).
        # So marginal should be negative? Or positive?
        # Actually, let's just rely on the fact that if a constraint is active (slack ~ 0), it limits the radius.
        # We can compute forces based on active constraints without explicit duals if needed, 
        # but duals are better. 
        # In scipy.optimize.linprog (highs), res.ineqlin.marginals contains the dual variables.
        # For min problem, duals for <= constraints are <= 0? 
        # Let's verify logic: We want to MAXIMIZE sum radii. 
        # Our LP minimizes -sum radii.
        # If we relax a constraint (increase RHS), we can achieve a lower minimum (more negative), so higher sum.
        # So shadow price (sensitivity of objective to RHS) should be negative.
        # But scipy usually returns positive marginals for active constraints in min problem? 
        # Let's just use the magnitude or handle signs carefully.
        # Actually, simpler heuristic: Identify active constraints by checking slack.
        
        slack = b_ub - A_ub @ radii
        active_threshold = 1e-9
        active_mask = slack < active_threshold
        
        # We return radii, sum, and mask of active constraints to compute forces
        # Mapping active constraints back to types
        # 0 to 4n-1: boundaries
        # 4n to end: distances
        
        active_indices = np.where(active_mask)[0]
        
        return radii, -res.fun, active_indices
    else:
        # Fallback if LP fails, return small radii
        return np.full(n, 1e-6), 1e-6, []

def compute_forces(centers, active_indices, n):
    """
    Compute forces on centers based on active constraints.
    Returns: force_array of shape (n, 2)
    """
    forces = np.zeros((n, 2))
    
    # Constants for boundary and distance constraints mapping
    num_boundary = 4 * n
    boundary_indices = range(num_boundary)
    dist_indices = range(num_boundary, num_boundary + n * (n - 1) // 2)
    
    # Identify which type of constraint each active index is
    for idx in active_indices:
        if idx < num_boundary:
            # Boundary constraint
            # Determine which circle and which side
            i = idx // 4
            side = idx % 4
            # side 0: r_i <= x_i  (Left wall, x_i is small) -> Force +x
            # side 1: r_i <= 1-x_i (Right wall, x_i is large) -> Force -x
            # side 2: r_i <= y_i  (Bottom wall) -> Force +y
            # side 3: r_i <= 1-y_i (Top wall) -> Force -y
            
            # Heuristic force strength
            strength = 1.0 
            
            if side == 0:
                forces[i, 0] += strength
            elif side == 1:
                forces[i, 0] -= strength
            elif side == 2:
                forces[i, 1] += strength
            elif side == 3:
                forces[i, 1] -= strength
                
        else:
            # Distance constraint
            # Map index back to pair (i, j)
            # Index in dist list: k = idx - num_boundary
            # Pairs are generated as (0,1), (0,2)...(0,n-1), (1,2)...
            # We can reconstruct or just iterate? 
            # Faster to just check all pairs? No, active_indices is small?
            # Actually, reconstructing (i,j) from index is O(1) or we can just iterate all pairs and check if active?
            # Since N=26, n*(n-1)/2 = 325. Iterating is fine.
            pass

    # Re-do force calculation for distances by checking all pairs against active_indices set
    # This avoids complex index math errors
    active_set = set(active_indices)
    offset = num_boundary
    
    # Precompute pair index mapping or just iterate
    # Let's just iterate pairs, it's fast enough
    # To map (i, j) to index:
    # The order was: for i in 0..n-1: for j in i+1..n-1
    # We can compute index formula or just build a map.
    # Let's build a map once? No, inside function is fine.
    
    # Actually, simpler: Iterate all pairs, calculate index, check if active.
    # But calculating index formula:
    # For pair (i, j) with i < j:
    # Number of pairs before row i: sum_{k=0}^{i-1} (n - 1 - k) = i*n - i*(i+1)/2
    # Plus (j - (i + 1))
    # Index = i*n - i*(i+1)/2 + j - i - 1 + offset
    
    # Let's verify for i=0, j=1: 0 - 0 + 1 - 0 - 1 + offset = offset. Correct.
    # i=0, j=2: 0 - 0 + 2 - 0 - 1 + offset = offset + 1. Correct.
    # i=1, j=2: 1*n - 1 + 2 - 1 - 1 + offset = n - 1 + offset.
    # Pairs before row 1 (i=0) are n-1 pairs (0,1)...(0,n-1).
    # So index should be (n-1) + offset.
    # Formula: 1*n - 1 + 2 - 1 - 1 = n - 1. Correct.
    
    # To optimize, we can just check active indices and deduce pairs?
    # Or just iterate pairs. 325 iterations is nothing.
    
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate index
            idx = i * n - (i * (i + 1)) // 2 + j - i - 1 + offset
            if idx in active_set:
                # Distance constraint active: r_i + r_j = dist(i, j)
                # We want to increase distance.
                # Vector from j to i
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist > 1e-12:
                    unit_vec = vec / dist
                    # Push i away from j, j away from i
                    # Strength proportional to... maybe 1/dist or constant?
                    # Constant is fine.
                    strength = 1.0 
                    forces[i] += unit_vec * strength
                    forces[j] -= unit_vec * strength
                    
    return forces

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    
    # 1. Initialization
    # Try to pack in a hexagonal-ish pattern or grid
    # A 5x5 grid has 25 spots. We need 26.
    # Let's try to fill a 5x5 grid and add one in the middle or perturb.
    # Or just random perturbation of a dense packing.
    
    # Let's use a grid with some randomness
    # 5 rows, 5 cols -> 25 circles.
    # We need 26. Maybe 6 in one row?
    # Or just spread them out.
    # Let's try a spiral or just random with repulsion?
    # Let's start with a 5x5 grid and add one at center, then let optimizer fix it.
    
    # Grid positions
    # Spacing 1/5 = 0.2. Centers at 0.1, 0.3, 0.5, 0.7, 0.9
    grid_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    count = 0
    for r in grid_coords:
        for c in grid_coords:
            if count < n:
                centers[count] = [c, r]
                count += 1
    
    # If we filled 25, add 26th at center?
    if count < n:
        centers[count] = [0.5, 0.5]
        count += 1
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    
    # 2. Optimization Loop
    num_iterations = 2000
    step_size = 0.05
    
    for it in range(num_iterations):
        # Solve LP
        radii, current_sum, active_indices = solve_lp_for_radii(centers)
        
        # Compute forces
        forces = compute_forces(centers, active_indices, n)
        
        # Update centers
        # Scale forces?
        # If forces are large, maybe dampen.
        # Also ensure centers stay in [0, 1]
        
        # Adaptive step size?
        # Reduce step size over time
        current_step = step_size * (1.0 / (1.0 + it * 0.01))
        
        centers += forces * current_step
        
        # Clip to [0, 1]
        centers = np.clip(centers, 0.0, 1.0)
        
        # Optional: If progress is slow, maybe try to randomize?
        # But forces should handle it.
        
    # 3. Final LP solve to get exact radii
    radii, final_sum, _ = solve_lp_for_radii(centers)
    
    # 4. Validation
    # Ensure radii are non-negative (LP should guarantee, but numerical issues?)
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic check
    # print(centers)
    # print(radii)

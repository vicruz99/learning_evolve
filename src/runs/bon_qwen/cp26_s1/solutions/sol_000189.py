# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9afef83a) state=4a5b04d9 sum of radii=1.890818 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # ---------------------------------------------------------
    # Phase 1: Initialization with a Hexagonal Lattice Structure
    # ---------------------------------------------------------
    centers = np.zeros((n, 2))
    
    # We want to fit 26 circles. A hexagonal pattern is denser.
    # Let's try row lengths that sum to 26.
    # A pattern like 5-6-5-6-4 (26 circles) works well in a hex packing.
    # Row counts
    row_counts = [5, 6, 5, 6, 4]
    
    # Initial guess for radius (smaller than 0.1 to allow optimization room)
    r_guess = 0.08
    
    idx = 0
    # Vertical spacing for hex packing
    dy = r_guess * np.sqrt(3)
    y_curr = r_guess
    
    for i, count in enumerate(row_counts):
        # Horizontal spacing
        dx = 2 * r_guess
        
        # Calculate x range to center the row within [0, 1]
        # The row width is (count - 1) * dx
        row_width = (count - 1) * dx
        
        # Center the row in the unit square
        # The centers should be within [r, 1-r]
        # Start x = (1 - row_width) / 2 + r_guess ?
        # Actually, simpler: span the available width [r, 1-r]
        # But for hex, we often shift.
        # Let's just space them evenly in [r_guess, 1-r_guess] first for the initialization logic
        # Then we will refine.
        
        # To allow optimization to work, let's place them in a dense valid configuration.
        # Let's calculate max possible r for this specific layout to start close to optimal.
        
        # For now, place them linearly in a row
        x_start = r_guess
        x_end = 1 - r_guess
        
        if count > 1:
            step = (x_end - x_start) / (count - 1)
        else:
            step = 0
            x_start = 0.5 # Center single circle
            
        for j in range(count):
            x = x_start + j * step
            
            # Apply hexagonal shift for odd rows (1-indexed in list, so i=1, 3...)
            if i % 2 == 1:
                x += dx / 2 # Shift by radius amount roughly
            
            # Ensure boundaries are respected for the initial placement
            x = max(r_guess, min(1 - r_guess, x))
            
            centers[idx] = [x, y_curr]
            idx += 1
        
        y_curr += dy

    # ---------------------------------------------------------
    # Phase 2: Optimization
    # ---------------------------------------------------------
    # We will optimize both centers and radii.
    # Objective: Maximize sum(radii)
    # Constraints:
    # 1. x_i - r_i >= 0
    # 2. x_i + r_i <= 1
    # 3. y_i - r_i >= 0
    # 4. y_i + r_i <= 1
    # 5. dist(i, j) >= r_i + r_j
    
    # This is a non-convex problem. We can use a local search.
    # Let's define a function that, given centers, computes the max possible radii.
    # Then we can optimize centers to maximize sum of these radii.
    
    def get_max_radii(centers_):
        """
        Given fixed centers, compute the maximum possible radius for each circle
        such that they don't overlap and stay in bounds.
        This is essentially finding the distance to the nearest neighbor and boundary.
        """
        radii = np.full(n, 1.0) # Start large
        
        # Constraint from boundaries
        for i in range(n):
            x, y = centers_[i]
            dist_to_bound = min(x, 1-x, y, 1-y)
            radii[i] = dist_to_bound
            
        # Constraint from neighbors
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers_[i] - centers_[j])**2))
                # dist >= r_i + r_j => r_i <= dist - r_j
                # We need to satisfy this for all pairs.
                # A simple way is to iteratively clamp or solve a system.
                # But for a quick estimate: r_i = min(r_i, (dist - r_j)) is not quite right because r_j is also variable.
                # Actually, if centers are fixed, the problem is finding max r_i such that r_i + r_j <= dist.
                # This is equivalent to r_i <= dist - r_j.
                # This looks like a system of inequalities. 
                # A safe upper bound for r_i is min_j (dist_ij / 2) ? No.
                # If we assume all radii are equal, r <= min(dist)/2.
                # But radii can differ.
                # However, to maximize sum, we usually want them somewhat equal.
                # Let's use a relaxation: r_i <= dist_ij - r_j.
                # This is hard to solve directly in one pass.
                # Let's just compute the "clearance" for each circle assuming neighbors have radius 0? No.
                # Let's assume equal radii for this step to get a scalar, then distribute?
                # Actually, the validation allows different radii.
                
                # Let's try a simpler approach:
                # The radius of circle i is limited by half the distance to any other circle j IF r_i = r_j.
                # But if r_j is small, r_i can be larger.
                # However, since we want to maximize SUM, and the packing is dense, 
                # it's highly likely that optimal radii are close to equal.
                # Let's estimate r_i based on min distance to any other center, assuming that neighbor also wants to be big.
                # r_i approx min( dist_ij / 2 ).
                
                # Better heuristic: 
                # r_i = min( dist_to_bound, min_j( dist_ij / 2 ) )
                # This is a lower bound on the optimal equal radius.
                pass

        # Let's do a proper calculation.
        # We can solve for r_i.
        # But for the optimizer, let's just return a valid set of radii.
        # A simple valid set: r_i = min( dist_to_bound_i, min_{j!=i} (dist_ij / 2) )
        # This guarantees no overlap if all radii are set this way?
        # If r_i = dist_ij / 2 and r_j = dist_ij / 2, then r_i + r_j = dist_ij. OK.
        # If r_i is limited by another k, r_i < dist_ij / 2, then r_i + r_j < dist_ij. OK.
        # So this formula gives a valid packing.
        
        radii = np.zeros(n)
        for i in range(n):
            # Boundary constraint
            max_r = min(centers_[i][0], 1 - centers_[i][0], centers_[i][1], 1 - centers_[i][1])
            
            # Neighbor constraint
            min_dist = np.inf
            for j in range(n):
                if i == j: continue
                d = np.sqrt(np.sum((centers_[i] - centers_[j])**2))
                if d < min_dist:
                    min_dist = d
            
            radii[i] = min(max_r, min_dist / 2.0)
            
        return radii

    # We want to maximize sum(radii) = sum( min(max_r, min_dist/2) )
    # This function is non-smooth (min operations).
    # But we can use a smooth approximation or just rely on the optimizer to find good centers.
    # Alternatively, we can optimize the sum of min_distances directly.
    # Maximizing min_distance is the standard "MaxMin" problem.
    # If we maximize min_distance, we maximize the equal radius packing.
    # Then we can try to adjust radii.
    
    # Let's optimize centers to maximize the minimum distance between any pair and to boundaries.
    # Let f(c) = min( min_i(dist_to_bound_i), min_{i<j}(dist_ij/2) )
    # Maximize f(c). Then set all radii = f(c).
    # This gives a baseline.
    
    # Then we can try to "grow" specific circles if they have more space.
    
    # Helper for the optimizer
    def objective(centers_flat):
        c = centers_flat.reshape(n, 2)
        
        # Penalize out of bounds
        penalty = 0
        for i in range(n):
            x, y = c[i]
            if x < 0 or x > 1 or y < 0 or y > 1:
                penalty -= 1000 * max(0, -x, x-1, -y, y-1)
        
        if penalty < -100: return penalty # Infeasible
        
        # Calculate min distance to boundary for all
        dists_bound = []
        for i in range(n):
            x, y = c[i]
            d_b = min(x, 1-x, y, 1-y)
            dists_bound.append(d_b)
        
        # Calculate min distance between pairs
        dists_pair = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                dists_pair.append(d)
        
        all_dists = dists_bound + [d/2 for d in dists_pair]
        min_dist = min(all_dists)
        
        # We want to maximize this min_dist.
        # To make it smoother for optimization, we can use a soft min or just return negative min_dist.
        return -min_dist

    # Optimize centers
    # Initial centers are already reasonable.
    # We will run a few iterations of Nelder-Mead or similar.
    x0 = centers.flatten()
    
    # Bounds for centers: [0, 1]
    bounds = [(0, 1)] * (2 * n)
    
    # Use a robust optimizer
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-9})
    
    optimized_centers = res.x.reshape(n, 2)
    
    # Now compute radii based on these optimized centers.
    # Since we optimized for min-distance, all circles can potentially have the same radius.
    # However, to maximize sum, we might be able to increase some radii if they are not the bottleneck.
    # But with a symmetric dense packing, they will likely be equal.
    # Let's compute the valid radii for this configuration.
    
    final_radii = get_max_radii(optimized_centers)
    
    # Check if we can do better with non-equal radii logic directly?
    # The get_max_radii function already assigns the max possible radius to each given fixed centers.
    # So sum(final_radii) is the max sum for this center configuration.
    
    # One more refinement: The centers optimization targeted equal radii.
    # It's possible that shifting a center to allow one circle to grow significantly 
    # while shrinking another slightly might increase the sum.
    # But sum of radii is usually maximized when radii are equal in dense packings.
    # Let's stick with this result.
    
    # Ensure strict non-overlap with a small safety margin if needed, 
    # but the math should be exact.
    
    # Let's run a final validation logic to be safe and return.
    
    return optimized_centers, final_radii, np.sum(final_radii)

# Helper function for validation inside run_packing if needed, 
# but the problem asks to return the tuple. 
# The provided validate_packing is for the evaluator.

# Let's refine the get_max_radii to be more precise.
# The logic: r_i <= dist_ij - r_j is a system.
# If we just use r_i = min(dist_ij/2), it's a valid lower bound.
# Is it the maximum?
# If r_i = dist_ij/2 for all j, then r_i + r_j = dist_ij. Tight.
# If r_i is limited by boundary, r_i = dist_to_bound.
# If r_i < dist_ij/2 for some j, then r_i + r_j < dist_ij is satisfied if r_j is also constrained.
# Yes, setting r_i = min(dist_to_bound, min_j(dist_ij/2)) is a valid configuration.
# And it's likely near optimal for sum.

if __name__ == "__main__":
    centers, radii, total_r = run_packing()
    print(f"Sum of radii: {total_r}")
    print(f"Radii range: [{np.min(radii)}, {np.max(radii)}]")

# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b5cb09ab) state=41f59da2 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, dual_annealing

def calculate_sum_radii(centers):
    """
    Calculates the maximum possible radii for a given set of centers 
    such that circles do not overlap and stay within the unit square.
    This is solved approximately by checking distances and boundaries.
    For optimization, we use a simplified model where we assume a single 
    common radius r, and calculate the max r allowed by constraints.
    """
    n = centers.shape[0]
    # Check boundaries
    min_r = 1.0
    for i in range(n):
        x, y = centers[i]
        min_r = min(min_r, x, 1-x, y, 1-y)
    
    # Check inter-circle distances
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            # 2r <= dist => r <= dist / 2
            min_r = min(min_r, dist / 2.0)
            
    return min_r * n

def objective(centers_flat):
    """
    Objective function for optimization: minimize negative sum of radii.
    centers_flat is a 1D array of shape (52,) representing 26 (x,y) pairs.
    """
    centers = centers_flat.reshape(-1, 2)
    # We want to maximize sum of radii. 
    # Assuming equal radii r, sum = 26 * r.
    # Max r is limited by boundaries and pairwise distances.
    r = 1.0
    n = 26
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # Penalty for being outside or close to boundary
        # We want x >= r, x <= 1-r => 2r <= 2x and 2r <= 2(1-x)
        # But r is variable. 
        # Let's just compute the feasible r for these centers.
        r = min(r, x, 1-x, y, 1-y)
        
    # Distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            r = min(r, dist / 2.0)
            
    # Return negative sum to minimize
    return - (26 * r)

def run_packing():
    """
    Generates the packing for 26 circles.
    """
    # Initial heuristic: Hexagonal packing
    # 6 rows: 5, 4, 5, 4, 5, 3 circles
    # This totals 26 circles.
    # We need to fit them in [0,1]x[0,1].
    
    # Let's estimate a good radius r. 
    # For 26 circles, r is slightly less than 0.1.
    # Let's start with r = 0.09 and adjust.
    
    r_init = 0.092
    centers = []
    
    # Row configuration
    # Row 0: 5 circles, y = r
    # Row 1: 4 circles, y = r + r*sqrt(3), x shifted by r
    # Row 2: 5 circles, y = r + 2*r*sqrt(3)
    # Row 3: 4 circles, y = r + 3*r*sqrt(3), x shifted by r
    # Row 4: 5 circles, y = r + 4*r*sqrt(3)
    # Row 5: 3 circles, y = r + 5*r*sqrt(3), x centered?
    
    # Let's calculate vertical spacing
    dy = r_init * np.sqrt(3)
    
    # Row 0: 5 circles
    # x coords: r, 3r, 5r, 7r, 9r -> total width 10r? 
    # 10 * 0.092 = 0.92. Fits in 1.
    # But we can center them.
    # Width of 5 circles touching is 10r.
    # Available width 1. Slack = 1 - 10r.
    # Let's distribute slack.
    
    row_counts = [5, 4, 5, 4, 5, 3]
    
    current_y = r_init
    for row_idx, count in enumerate(row_counts):
        # Horizontal shift for staggered rows
        if row_idx % 2 == 1:
            x_start = r_init * 2 # Shift by r relative to grid? 
            # In hex packing, shift is r.
            # Base grid x: r, 3r, ...
            # Shifted: 2r, 4r, ...
            x_start = 2 * r_init
        else:
            x_start = r_init
            
        # Generate x coords for this row
        # If we just place them at x_start, x_start+2r, ...
        # We need to check if they fit.
        # Last x = x_start + (count-1)*2r.
        # Right edge = Last x + r.
        # Must be <= 1.
        
        # Let's calculate ideal positions based on r_init, then optimize.
        xs = []
        for k in range(count):
            x = x_start + k * 2 * r_init
            xs.append(x)
        
        for x in xs:
            centers.append([x, current_y])
            
        current_y += dy

    centers = np.array(centers)
    
    # Verify initial count
    assert len(centers) == 26
    
    # Initial bounds for optimization: [0, 1] for x and y
    bounds = [(0.0, 1.0) for _ in range(52)]
    
    # Use optimization to maximize radius
    # We optimize centers to maximize the minimum distance between circles and boundaries.
    # This is equivalent to maximizing the radius r of equal circles.
    
    # Helper for the optimizer
    def neg_max_r(x_flat):
        c = x_flat.reshape(-1, 2)
        # Clip to valid range to avoid huge penalties if out of bounds? 
        # But optimizer handles bounds.
        # Calculate min distance to boundary
        min_d = 1.0
        for i in range(26):
            x, y = c[i]
            min_d = min(min_d, x, 1-x, y, 1-y)
        
        # Calculate min distance between circles
        for i in range(26):
            for j in range(i + 1, 26):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                min_d = min(min_d, d / 2.0)
        
        return -min_d # Maximize min_d

    # Run optimization
    # Using dual_annealing for global search might be better, but slower.
    # Let's try Nelder-Mead or L-BFGS-B first from the heuristic start.
    
    x0 = centers.flatten()
    
    # Try local optimization first
    res = minimize(neg_max_r, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    
    # If result is good, maybe run a few random restarts or a global method?
    # Given the complexity, let's try to run dual_annealing if time permits, 
    # but for the solution block, we should rely on the optimized result.
    # Let's try a quick global search with few iterations or just rely on the local optimum
    # since the hex packing is a good basin.
    
    # Let's try to improve with a global optimizer like dual_annealing with limited evals
    # Actually, dual_annealing can be slow. 
    # Let's stick to the refined local optimum.
    
    optimal_centers = res.x.reshape(-1, 2)
    max_r = -res.fun
    
    # Calculate radii
    # If we optimized for equal radii, all radii are max_r.
    radii = np.full(26, max_r)
    
    # However, the problem allows variable radii. 
    # But as reasoned, equal radii is likely optimal for sum.
    # Let's verify if we can squeeze more by adjusting radii?
    # Actually, if we have optimal centers for equal radii r, 
    # the sum is 26*r. 
    # If we allow variable radii, we might get more?
    # But the bottleneck is the geometry.
    # Let's assume equal radii is the target strategy.
    
    # Refine: maybe the optimal packing for 26 circles has slightly different radii?
    # But for simplicity and robustness, equal radii is a strong candidate.
    
    # Let's check if we can simply scale up?
    # The optimization found the max r for equal circles.
    
    # One potential improvement: 
    # The 6th row has only 3 circles. Maybe we can fit 4?
    # 5+4+5+4+5+4 = 27 circles.
    # If we can fit 27, we can surely fit 26 with larger radius.
    # Let's try to fit 27 circles in hex pattern?
    # Rows: 5, 4, 5, 4, 5, 4.
    # Last row y = r + 5*sqrt(3)*r.
    # Top constraint: r + 5*sqrt(3)*r + r <= 1
    # r(2 + 5*sqrt(3)) <= 1
    # 5*sqrt(3) approx 8.66.
    # r(10.66) <= 1 => r <= 0.0938.
    # Width constraint for 4 circles (shifted row):
    # 4 circles width? 
    # Shifted row x: 2r, 4r, 6r, 8r.
    # Right edge 8r + r = 9r.
    # 9 * 0.0938 = 0.844. Fits.
    # So we can fit 27 circles with r ~ 0.0938.
    # Sum = 27 * 0.0938 = 2.53.
    # Wait, 26 circles with r ~ 0.096?
    # If we remove 1 circle, we can increase r.
    
    # Let's re-optimize for 26 circles specifically.
    # The previous config had row counts [5, 4, 5, 4, 5, 3].
    # Total 26.
    # The last row had 3 circles, which is less dense.
    # Maybe [5, 4, 5, 4, 5, 3] is not optimal.
    # Maybe [5, 4, 5, 4, 4, 4]? No, symmetry.
    # Maybe [4, 5, 4, 5, 4, 4]?
    
    # Let's just use the optimization result on the [5,4,5,4,5,3] config.
    # And maybe try [5,4,5,4,4,4] or [4,5,4,5,4,4] as alternative starts?
    # Actually, 5-4-5-4-5-3 sums to 26.
    # 5-4-5-4-4-4 sums to 26.
    # 4-5-4-5-4-4 sums to 26.
    
    # Let's try to run optimization on a few configurations and pick the best.
    
    best_sum = -1
    best_centers = None
    best_radii = None
    
    configs = [
        [5, 4, 5, 4, 5, 3],
        [5, 4, 5, 4, 4, 4],
        [4, 5, 4, 5, 4, 4],
        [5, 5, 4, 5, 4, 3], # Just guessing patterns
        [4, 4, 5, 5, 4, 4]
    ]
    
    # Helper to generate centers for a config
    def generate_config(row_counts, r_guess):
        c = []
        dy = r_guess * np.sqrt(3)
        y = r_guess
        for i, count in enumerate(row_counts):
            if i % 2 == 1:
                x_start = 2 * r_guess
            else:
                x_start = r_guess
            for k in range(count):
                x = x_start + k * 2 * r_guess
                c.append([x, y])
            y += dy
        return np.array(c)

    # Try a few random perturbations and optimize
    for config in configs:
        if sum(config) != 26: continue
        
        r_guess = 0.095 # Initial guess
        # Adjust r_guess to fit roughly
        # Height check: 2r + (rows-1)*sqrt(3)*r <= 1
        # r(2 + 5*1.732) <= 1 => r <= 1/10.66 = 0.0938
        r_guess = 0.090
        
        centers_init = generate_config(config, r_guess)
        
        # Optimize
        res = minimize(neg_max_r, centers_init.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000})
        current_sum = -res.fun * 26
        if current_sum > best_sum:
            best_sum = current_sum
            optimal_centers = res.x.reshape(-1, 2)
            
    # Final radii
    radii = np.full(26, -res.fun) # res.fun is negative min_d (which is r)
    
    # Just to be safe, re-calculate max feasible radii for the final centers
    # In case equal radii is not the absolute max sum (though likely is)
    # We can solve the LP for radii, but let's stick to equal for simplicity 
    # unless we can easily do better.
    # Actually, if we have optimal centers for equal circles, 
    # the sum is maximized.
    
    # However, the validation function checks for overlaps.
    # We need to ensure radii are valid.
    # The optimization maximizes r such that constraints hold.
    # So r = -res.fun is valid.
    
    # Let's add a tiny epsilon to be safe? No, constraints are strict.
    # The optimization finds the boundary.
    
    # One last check: 
    # The target is 2.636.
    # My estimated r ~ 0.0938 -> sum ~ 2.43.
    # This is still below 2.636.
    # Is it possible to reach 2.636?
    # 2.636 / 26 = 0.10138.
    # This requires r > 0.1.
    # As analyzed, r > 0.1 for 26 circles seems impossible for equal circles.
    # Therefore, variable radii might be required.
    
    # Let's try to construct a solution with variable radii.
    # Strategy: Pack 25 circles of radius 0.1 in 5x5 grid.
    # Then insert 1 circle in a gap?
    # We found gap radius ~0.0414.
    # Sum = 2.5 + 0.0414 = 2.5414.
    # Still low.
    
    # What if we have 4 circles of radius 0.2 in corners?
    # And fill the rest?
    # 4 circles of r=0.2 at (0.2, 0.2), (0.8, 0.2), ...
    # Distance between (0.2, 0.2) and (0.8, 0.2) is 0.6.
    # 2r = 0.4. 0.6 > 0.4. OK.
    # Sum = 0.8.
    # Remaining space is complex.
    
    # Maybe the target 2.636 is a loose target and 2.4-2.5 is acceptable?
    # But the prompt says "Target: 2.636".
    # I should try to get as close as possible.
    
    # Let's try to optimize variable radii directly?
    # This is harder.
    
    # Alternative: 
    # Maybe I can pack 26 circles with radii varying slightly.
    # Let's use the equal circle optimization result, 
    # but then try to "inflate" radii where possible.
    # Actually, if the optimization finds the max r for EQUAL circles,
    # it finds the configuration where the "bottleneck" constraints are tight.
    # If we allow variable radii, we might relax some bottlenecks?
    # No, bottlenecks are geometric.
    
    # Let's double check the row counts.
    # Maybe [5, 5, 5, 5, 5, 1]?
    # 5 rows of 5.
    # Height for 5 rows: 2r + 4*sqrt(3)r = r(2+6.92) = 8.92r.
    # r <= 1/8.92 = 0.112.
    # Width for 5 circles: 10r.
    # r <= 0.1.
    # So width is limiting. r=0.1.
    # Sum = 26 * 0.1 = 2.6.
    # Wait. 5 rows of 5 circles is 25 circles.
    # Plus 1 circle.
    # If we fit 25 circles of r=0.1, we have 1 circle left.
    # Where can we put it?
    # In 5x5 grid, gaps are small.
    # But maybe we can rearrange?
    
    # If we use 5 rows of 5 circles, r=0.1.
    # Height used: 2*0.1 + 4*0 = 0.2? No, square grid.
    # Square grid: 5 rows, spacing 0.2.
    # y coords: 0.1, 0.3, 0.5, 0.7, 0.9.
    # Fits perfectly.
    # We have 25 circles.
    # We need 26.
    # Can we add a 26th circle?
    # In the center of the square? (0.5, 0.5).
    # Occupied by a circle.
    # In the gaps?
    # Gap at (0.2, 0.2) relative to grid?
    # Grid points are (0.1+2i*0.1, 0.1+2j*0.1)? No, spacing 0.2.
    # Points: (0.1, 0.1), (0.3, 0.1), ...
    # Gaps at (0.2, 0.2).
    # Distance to neighbors (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3) is sqrt(0.01+0.01)=0.1414.
    # Radius 0.1.
    # Gap radius 0.0414.
    # If we put a circle of radius 0.0414, sum = 2.5414.
    
    # Is there a better arrangement?
    # Hexagonal packing of 26 circles.
    # We found r ~ 0.0938 for 27 circles.
    # For 26 circles, r might be slightly larger.
    # Let's optimize the [5,4,5,4,5,3] config again carefully.
    # Maybe r can reach 0.10?
    # If r=0.1, width for 5 circles is 1.0. Fits exactly.
    # Height for 6 rows: 2r + 5*sqrt(3)r = 0.2 + 5*1.732*0.1 = 0.2 + 0.866 = 1.066.
    # Too tall.
    # So for 6 rows, r must be smaller.
    # What if we use 5 rows?
    # 5, 5, 5, 5, 6? No.
    # 5 rows of 5 is 25.
    # 5 rows with 5, 5, 5, 5, 6?
    # 6 circles in a row require width 12r.
    # 12*0.1 = 1.2. Too wide.
    
    # Maybe 5 rows: 5, 5, 5, 5, 6 is impossible.
    # Maybe 5 rows: 5, 5, 5, 5, 5 is 25.
    # Maybe 4 rows: 7, 6, 7, 6?
    # 7 circles width 14r. Impossible.
    
    # It seems 26 circles with r=0.1 is impossible.
    # So sum < 2.6 is likely for equal circles.
    
    # However, the target is 2.636.
    # Maybe I can use variable radii to get higher sum.
    # If I have 25 circles of radius 0.1 (sum 2.5) and 1 circle of radius 0.1?
    # No, can't fit 26th of radius 0.1.
    # But maybe 24 circles of 0.1 and 2 larger?
    # No, space is limited.
    
    # Let's try to optimize the sum of radii directly with variable radii.
    # We can use a simple iterative method:
    # 1. Place 26 circles randomly or in grid.
    # 2. Solve LP for max radii.
    # 3. Adjust centers to improve.
    
    # Given the constraints of the environment, I'll provide the best equal circle packing I can find,
    # as it is a solid baseline. 
    # But I will try to push it.
    
    # Re-evaluating the [5, 4, 5, 4, 5, 3] config.
    # With r=0.095, sum = 2.47.
    # Can we do better?
    # What if we tilt the lattice?
    
    # Let's stick to the optimized centers from the code block.
    
    # Wait, I can use the `minimize` result to get the centers.
    # Then I can compute the radii.
    # If the optimizer found r=0.096, sum=2.496.
    
    # Let's check if there is a known better packing.
    # "26 circles in a square".
    # Some sources say r=0.0905 for equal circles.
    # But maybe variable radii allows more.
    
    # Let's try to return the result from the optimization.
    
    return optimal_centers, radii, sum(radii)

# Execution logic to find the best packing
def run_packing():
    # Define the objective function again for clarity
    def neg_max_r(x_flat):
        c = x_flat.reshape(-1, 2)
        min_d = 1.0
        for i in range(26):
            x, y = c[i]
            min_d = min(min_d, x, 1-x, y, 1-y)
        for i in range(26):
            for j in range(i + 1, 26):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                min_d = min(min_d, d / 2.0)
        return -min_d

    bounds = [(0.0, 1.0) for _ in range(52)]
    
    best_sum = -1
    best_res = None
    
    # Try multiple starting configurations
    configs = [
        [5, 4, 5, 4, 5, 3],
        [5, 4, 5, 4, 4, 4],
        [4, 5, 4, 5, 4, 4],
        [5, 5, 4, 5, 4, 3]
    ]
    
    for config in configs:
        if sum(config) != 26: continue
        
        # Estimate r to fit roughly
        # Height: 2r + (rows-1)*sqrt(3)*r <= 1
        # Width: max_row_count * 2r (approx) <= 1
        # Simple guess r = 0.09
        r_guess = 0.09
        
        centers = []
        dy = r_guess * np.sqrt(3)
        y = r_guess
        for i, count in enumerate(config):
            if i % 2 == 1:
                x_start = 2 * r_guess
            else:
                x_start = r_guess
            for k in range(count):
                x = x_start + k * 2 * r_guess
                centers.append([x, y])
            y += dy
            
        centers_arr = np.array(centers)
        
        # Optimize
        res = minimize(neg_max_r, centers_arr.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
        
        current_sum = -res.fun * 26
        if current_sum > best_sum:
            best_sum = current_sum
            best_res = res
            
    optimal_centers = best_res.x.reshape(-1, 2)
    radii = np.full(26, -best_res.fun)
    
    return optimal_centers, radii, sum(radii)

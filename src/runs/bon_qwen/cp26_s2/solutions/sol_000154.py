# sol_000154 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 46a34d55) state=5ec801a5 sum of radii=2.453553 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a hybrid approach: initialize with a dense hexagonal lattice,
    then use a simple iterative expansion to maximize radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 1. Initialization: Hexagonal Lattice
    # We arrange circles in rows.
    # To fit 26, we can use 5 rows of 5 and 1 extra, or a staggered pattern.
    # A 6-row staggered pattern (5-4-5-4-5-3) is dense but might limit radius.
    # Let's try a 5-row pattern with 6 circles in one row? No, width issue.
    # Let's use 5 rows of 5, and place the 26th in the center gap?
    # Actually, a 5x5 grid is tight.
    # Let's try a configuration that allows slightly larger radii by not being a perfect grid.
    
    # Let's use a random initialization with rejection sampling or a grid,
    # then optimize.
    # A good grid for 26 circles:
    # 5 rows. 
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 6 circles (shifted) -> This is the key. If we can fit 6 in a shifted row, 
    # we might gain space.
    # But 6 circles need width.
    # Let's try 5 rows with counts [5, 5, 5, 5, 6] is impossible.
    # How about [5, 5, 5, 6, 5]?
    # If we shift the row with 6 circles, maybe it fits?
    # Centers of 6 circles: x1...x6.
    # If shifted, they fit in the gaps of 5 circles.
    # Gap width between 5 circles of radius r is 0 (touching).
    # So we need to expand the row below/above.
    
    # Let's use a simpler strategy:
    # Generate a hexagonal grid of 30 points, select the best 26 that fit in square.
    # Then optimize radii.
    
    # Hexagonal grid parameters
    # We want to cover [0,1]x[0,1].
    # Let's use a spacing s.
    # Points: (i*s + (j%2)*s/2, j*s*sqrt(3)/2)
    # We need to find s such that we have >= 26 points inside.
    # And then we can adjust.
    
    # Let's try s = 0.2.
    # j=0: y=0. x=0, 0.2, 0.4, 0.6, 0.8, 1.0. (6 points)
    # j=1: y=0.1732. x=0.1, 0.3, 0.5, 0.7, 0.9. (5 points)
    # j=2: y=0.3464. x=0, 0.2, 0.4, 0.6, 0.8, 1.0. (6 points)
    # j=3: y=0.5196. x=0.1, 0.3, 0.5, 0.7, 0.9. (5 points)
    # j=4: y=0.6928. x=0, 0.2, 0.4, 0.6, 0.8, 1.0. (6 points)
    # j=5: y=0.8660. x=0.1, 0.3, 0.5, 0.7, 0.9. (5 points)
    # j=6: y=1.0392. Out.
    # Total points: 6+5+6+5+6+5 = 33 points.
    # We need 26. We can select the central 26 or the ones that allow largest radii.
    # But these points are fixed. Radii would be limited by spacing.
    # Spacing s=0.2 => min dist 0.2. So r <= 0.1.
    # This gives sum <= 2.6.
    # We need > 2.636.
    # So we need spacing > 0.2.
    # If s > 0.2, we get fewer points.
    # s=0.22.
    # j=0: 0, 0.22, 0.44, 0.66, 0.88. (5 points). 1.1 out.
    # j=1: y=0.19. x=0.11, 0.33, 0.55, 0.77, 0.99. (5 points).
    # j=2: y=0.38. x=0, 0.22, 0.44, 0.66, 0.88. (5 points).
    # j=3: y=0.57. x=0.11, 0.33, 0.55, 0.77, 0.99. (5 points).
    # j=4: y=0.76. x=0, 0.22, 0.44, 0.66, 0.88. (5 points).
    # j=5: y=0.95. x=0.11, 0.33, 0.55, 0.77, 0.99. (5 points).
    # Total 30 points.
    # Spacing 0.22 => r <= 0.11.
    # Sum = 26 * 0.11 = 2.86.
    # But we need to check if they fit in the square.
    # The points must be at least r away from boundary.
    # For s=0.22, r=0.11.
    # Point (0,0) is invalid (needs r).
    # So we must shift the grid.
    # Shift x by r, y by r.
    # New x range: [0.11, 0.11 + 4*0.22] = [0.11, 0.99]. Fits in [0,1] if r=0.11?
    # 0.11 - 0.11 = 0. 0.99 + 0.11 = 1.1 > 1.
    # So (0.99, y) is too close to right wall.
    # We need to shrink or select points.
    
    # This suggests we can't simply use a grid with r=0.11.
    # We need to select 26 points that allow r ~ 0.101.
    
    # Let's try to construct a solution manually with 5 rows.
    # Row 0: 5 circles.
    # Row 1: 5 circles.
    # Row 2: 6 circles? No.
    # Row 2: 5 circles.
    # Row 3: 5 circles.
    # Row 4: 5 circles.
    # Total 25.
    # We need 1 more.
    # Place it in the center?
    # Center (0.5, 0.5).
    # Distance to nearest circle (e.g. at 0.5, 0.1? No, 0.5, 0.4?)
    # If we have 5 rows, y spacing is roughly 0.25.
    # Rows at 0.125, 0.375, 0.625, 0.875? No, 4 gaps.
    # 5 rows => 4 gaps.
    # y = 0.1, 0.3, 0.5, 0.7, 0.9.
    # Center of square is (0.5, 0.5).
    # If we have a circle at (0.5, 0.5), it overlaps with others?
    # If others are at (0.5, 0.5) in row 2?
    # Row 2 has 5 circles: 0.1, 0.3, 0.5, 0.7, 0.9.
    # So (0.5, 0.5) is occupied.
    # We need to move the 26th circle to a gap.
    # Gap between (0.3, 0.5) and (0.7, 0.5)? No, (0.5, 0.5) is there.
    # Gap between rows?
    # Between row 1 (y=0.3) and row 2 (y=0.5).
    # Gap y=0.4.
    # Circles at x=0.1, 0.3, 0.5, 0.7, 0.9 in both rows?
    # If aligned, no gap.
    # If shifted (hexagonal), gaps exist.
    # Row 1: 0.1, 0.3, 0.5, 0.7, 0.9.
    # Row 2: 0.2, 0.4, 0.6, 0.8, 1.0? (Shifted by 0.1).
    # If shifted by 0.1 (which is r if r=0.1? No, r=0.1, shift=r).
    # Then gaps are at x=0.15, 0.35, 0.55, 0.75?
    # Distance from (0.15, 0.4) to (0.1, 0.3)?
    # dx=0.05, dy=0.1. Dist = sqrt(0.0025+0.01) = 0.111.
    # Sum of radii = 0.1 + 0.1 = 0.2.
    # 0.111 < 0.2. Overlap.
    # So we can't just insert a circle of radius 0.1.
    # We need to reduce radius of existing circles to make space.
    
    # This brings us back to: equal circles with r < 0.1.
    # But we need sum > 2.636.
    # This implies we need UNEQUAL circles.
    # Some circles must be larger than 0.1, some smaller.
    # Where can we fit a larger circle?
    # In the corners?
    # If we have a large circle in corner (0.2, 0.2) with r=0.2?
    # It takes up a lot of space.
    # Maybe 4 large circles in corners?
    # r=0.2. Centers (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8).
    # Distance between (0.2, 0.2) and (0.8, 0.2) is 0.6.
    # Sum of radii 0.4. 0.6 > 0.4. OK.
    # Distance diagonal 0.848. Sum 0.4. OK.
    # These 4 circles leave a cross shape in the middle.
    # Width of cross: 1 - 2*0.2 = 0.6.
    # We need to fit 22 more circles in the remaining space.
    # The space is roughly a 0.6x0.6 square with 4 quarter-circles cut out?
    # Or just the central region.
    # In a 0.6x0.6 square, we can fit circles of radius ~0.1?
    # 6x6 grid of 0.1 circles? No, 0.6 width fits 3 circles of diameter 0.2 (r=0.1).
    # So we can fit 3x3 = 9 circles of radius 0.1 in the center.
    # Total circles = 4 (large) + 9 (small) = 13.
    # We need 26.
    # We can fit more small circles.
    # In the regions between large circles and boundary?
    # The large circles are at 0.2. Boundary at 0.
    # Gap width 0.2.
    # We can fit circles of radius 0.1 in the gaps?
    # Gap between (0.2, 0.2) and (0, 0.2)?
    # Region x in [0, 0.2].
    # We can fit a row of circles along the edge.
    # Radius 0.1.
    # How many? Width 0.2 fits 1 circle of diameter 0.2.
    # So 1 circle per side?
    # 4 sides => 4 circles.
    # Total 4+9+4 = 17.
    # Still not 26.
    # We can fit smaller circles.
    # If we use r=0.05 in the gaps?
    # Sum contribution is low.
    
    # This strategy (4 large + many small) is inefficient for sum of radii.
    # Because large circles consume area quadratically.
    # To maximize sum, we want radii to be as uniform as possible.
    
    # So we are back to: Why is target 2.636?
    # Maybe my 5x5 grid estimate is wrong.
    # 5x5 grid, r=0.1. Sum 2.5.
    # Can we increase r?
    # If we don't use a grid?
    # If we use a hexagonal packing of 26 circles?
    # We found r ~ 0.094. Sum 2.44.
    # This is WORSE than 5x5 grid.
    # So square packing is better for N=25.
    # For N=26, we add 1 circle.
    # We must reduce r from 0.1.
    # How much?
    # If we add 1 circle to 5x5, we need to create a gap.
    # The gap in 5x5 is 0.
    # We need to expand the grid.
    # New width W > 1? No, W=1 fixed.
    # So we must reduce r.
    # If we reduce r, sum decreases.
    # So N=26 with equal circles is worse than N=25?
    # Yes, sum of radii for N=25 is 2.5.
    # For N=26, r must be < 0.1.
    # So sum < 2.6.
    # BUT we have the 26th circle.
    # Sum = 25*r_new + r_new = 26*r_new.
    # If r_new = 0.099, sum = 2.574.
    # Still < 2.636.
    # So EQUAL circles cannot reach 2.636.
    
    # Therefore, the solution MUST use unequal circles.
    # And specifically, it must use a configuration where some circles are larger than 0.1.
    # How?
    # By having a configuration that is NOT a 5x5 grid.
    # Maybe a 6x4 grid?
    # 6 rows, 4 columns.
    # Width 4 circles => 4*d <= 1 => d <= 0.25. r <= 0.125.
    # Height 6 rows.
    # If square packing: 6*d <= 1 => d <= 0.166.
    # If hexagonal packing: Height = 2r + 5*sqrt(3)r = r(2 + 8.66) = 10.66r.
    # 10.66r <= 1 => r <= 0.0938.
    # So hexagonal 6x4 is bad (r=0.094).
    # Square 6x4 is bad (r=0.083).
    # What about 5 rows, 6 columns?
    # Width 6 circles => 6*d <= 1 => d <= 0.166. r <= 0.083.
    # Bad.
    
    # What about a configuration with 5 circles in some rows and 4 in others?
    # We analyzed this. 5-circle rows limit r to 0.1.
    # 4-circle rows allow r up to 0.125.
    # If we have mostly 4-circle rows, we can have larger r.
    # But we need 26 circles.
    # 7 rows of 4 = 28 circles.
    # We can take 26 circles from a 7x4 grid?
    # 7 rows.
    # Height constraint for 7 rows with r=0.125?
    # Square: 7 * 0.25 = 1.75. No.
    # Hexagonal: 6 * sqrt(3) * 0.125 + 0.25 = 1.3. No.
    # We need smaller r.
    # If we use hexagonal packing for 7 rows of 4.
    # H = r(2 + 6*1.732) = r(12.39).
    # 12.39r <= 1 => r <= 0.0807.
    # Sum = 26 * 0.0807 = 2.09.
    
    # It seems impossible to get sum > 2.6 with these simple grids.
    # UNLESS...
    # The circles are not arranged in rows.
    # Maybe a spiral?
    # Or maybe the target 2.636 is for a different N?
    # No, N=26.
    # Maybe I am missing a very dense packing.
    # Let's check the area again.
    # Sum radii = 2.636.
    # If equal, r = 0.1014.
    # Area = 26 * pi * (0.1014)^2 = 26 * 3.1416 * 0.01028 = 0.840.
    # Density 0.84.
    # This is feasible.
    # The problem is the boundary.
    # With density 0.84, we need to pack 26 circles of diameter 0.2028.
    # Total length of 5 circles = 1.014.
    # So we can't fit 5 in a row.
    # So we must fit them in a way that doesn't use 5 in a straight line.
    # e.g. 2 circles, gap, 3 circles?
    # Or a 2D lattice that is not aligned with axes.
    # A rotated square lattice?
    # If we rotate the 5x5 grid by 45 degrees?
    # Diagonal of square is 1.414.
    # 5x5 grid diagonal length = 4 * sqrt(2) * d?
    # Distance from corner to corner in 5x5 grid is 4 * 2r * sqrt(2) = 8r * 1.414 = 11.3r.
    # 11.3r <= 1.414 => r <= 0.125.
    # But the grid must fit inside the square.
    # A rotated grid is harder to fit.
    # However, it might allow larger r.
    # But usually, axis-aligned is optimal for square container.
    
    # Let's assume the target is achievable and write a solver.
    # I will use a multi-start local search.
    
    pass

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    import numpy as np
    
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # We will try several initial configurations and expand radii
    configs = []
    
    # Config 1: 5x5 Grid + 1
    # 5 rows, 5 cols.
    # x = 0.1, 0.3, 0.5, 0.7, 0.9
    # y = 0.1, 0.3, 0.5, 0.7, 0.9
    # 25 points. Add (0.5, 0.5) is overlap.
    # Add (0.2, 0.2)? Overlap.
    # Add (0.0, 0.0)? No.
    # Let's just perturb the 5x5 grid slightly to make room for 26th.
    # Or use a 6x4 grid with hexagonal spacing.
    
    # Config 2: Hexagonal 6x4 (24 points) + 2
    # s = 0.2
    # Points:
    pts = []
    s = 0.2
    # 6 rows
    for j in range(6):
        y = j * s * math.sqrt(3) / 2 + 0.1 # Shift to center
        shift = (j % 2) * s / 2
        for i in range(5): # 5 cols
            x = i * s + shift + 0.1
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
    # This might give more than 26 or less.
    # Let's just generate a random set of 26 points and optimize.
    
    # Better: Use a deterministic dense packing.
    # We'll use a "force directed" layout initialization.
    np.random.seed(42)
    # Start with random points
    # Then run a simple optimizer
    
    # Since I cannot run a loop in the return, I must compute it inside.
    # I will implement a simple gradient ascent on sum of radii.
    
    # Variables: x, y, r for each circle.
    # But r is determined by min(distances).
    # So we only optimize x, y.
    # Then r_i = min( x_i, 1-x_i, y_i, 1-y_i, min_{j!=i} dist(i,j)/2 )
    # Objective: sum(r_i).
    
    # This is a standard problem.
    # I'll use a simple iterative algorithm:
    # 1. Initialize centers randomly or in a grid.
    # 2. Compute radii.
    # 3. Move centers to increase radii.
    # 4. Repeat.
    
    # Let's use a grid initialization that is known to be good.
    # 5x5 grid is good for 25.
    # For 26, maybe a 5x5 grid with one circle pushed to a corner?
    # Or a 6x5 grid with some removed?
    
    # Let's try to generate a 5x5 grid, then add a 26th point,
    # and then run a few steps of "repulsion" to optimize.
    
    centers = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
        [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3],
        [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5],
        [0.1, 0.7], [0.3, 0.7], [0.5, 0.7], [0.7, 0.7], [0.9, 0.7],
        [0.1, 0.9], [0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9],
        [0.5, 0.5] # Duplicate!
    ])
    # Replace last with a valid point, e.g. (0.0, 0.0) is invalid.
    # (0.2, 0.2) is too close.
    # Let's use (0.2, 0.8) ?
    # Let's just use a 6x5 grid of 30 points and pick 26?
    # No, let's use the 5x5 grid and perturb it.
    # Actually, let's use a known optimal packing for N=26 if I can recall it?
    # No.
    
    # I will use a simple optimization loop inside the function.
    # It's deterministic if I fix the seed and steps.
    
    # Initialize with 5x5 grid + 1 point at (0.2, 0.2) is bad.
    # Let's place 26 points on a 6x5 grid (30 points) and remove 4?
    # 6x5 grid:
    # x = 0.2, 0.4, 0.6, 0.8
    # y = 0.2, 0.4, 0.6, 0.8
    # This is 4x4 = 16 points.
    # We need 26.
    # 5x5 = 25.
    # Let's use 5x5 grid with spacing 0.2.
    # Points: (0.1, 0.1) ... (0.9, 0.9).
    # Add one more point. Where?
    # Maybe (0.05, 0.05)? No, r would be small.
    # Maybe (0.5, 0.0)? No.
    # Let's try to find a spot with max min-distance.
    # In 5x5 grid, the "holes" are at (0.2, 0.2) etc? No, (0.1,0.1) is a center.
    # Holes are at (0.2, 0.2)? Distance to (0.1,0.1) is 0.141.
    # Distance to (0.3,0.1) is 0.141.
    # So r = 0.07.
    # Sum = 25*0.1 + 0.07 = 2.57.
    
    # Let's try to perturb the 5x5 grid to increase the sum.
    # We can use scipy.optimize.
    
    from scipy.optimize import minimize
    
    # Initial centers: 5x5 grid
    x0 = []
    for i in range(5):
        for j in range(5):
            x0.append([0.1 + i*0.2, 0.1 + j*0.2])
    # Add one more point
    x0.append([0.2, 0.8]) # Just a guess
    
    x0 = np.array(x0)
    
    def objective(x):
        # x is 26*2 array
        c = x.reshape(n, 2)
        r = np.zeros(n)
        for i in range(n):
            # Distance to boundary
            rb = min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1])
            # Distance to other circles
            rc = np.min([np.linalg.norm(c[i] - c[j]) for j in range(n) if j != i]) / 2
            r[i] = min(rb, rc)
        return -np.sum(r) # Minimize negative sum
        
    def constraints(x):
        # No overlap? 
        # The objective handles radii calculation, so constraints are implicit?
        # No, we need to ensure radii are non-negative.
        # But min() ensures r >= 0.
        # We just need to ensure x in [0,1].
        # Bounds handle that.
        return []

    # Bounds: [0, 1] for all coordinates
    bounds = [(0, 1) for _ in range(n*2)]
    
    # We can try a few random restarts
    best_res = None
    best_val = float('inf')
    
    for trial in range(5):
        # Perturb initial centers
        x_start = x0.copy()
        if trial > 0:
            x_start += np.random.normal(0, 0.05, x_start.shape)
            x_start = np.clip(x_start, 0, 1)
            
        try:
            res = minimize(objective, x_start, bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except:
            pass
            
    if best_res is not None:
        centers = best_res.x.reshape(n, 2)
    else:
        centers = x0
        
    # Compute final radii
    radii = np.zeros(n)
    for i in range(n):
        rb = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        rc = np.min([np.linalg.norm(centers[i] - centers[j]) for j in range(n) if j != i]) / 2
        radii[i] = min(rb, rc)
        
    sum_radii = np.sum(radii)
    
    # Validate
    # Note: The validation function in prompt is for checking.
    # We assume our calculation is correct.
    
    return centers, radii, sum_radii

# Wait, I cannot use scipy inside the solution if it's not guaranteed to be fast or available in the specific environment?
# The prompt says "You can use scientific libraries like scipy".
# So it's allowed.
# However, `minimize` might get stuck in local minima.
# And the objective function is non-smooth (min of distances).
# L-BFGS-B might struggle.
# But it's the best I can do without a complex custom solver.
# I will add a smoothing or use a simpler method?
# Actually, for this specific problem, a hardcoded high-quality solution is better.
# But I don't have it.
# So I'll stick with the optimization.
# I'll make sure to seed random for reproducibility.

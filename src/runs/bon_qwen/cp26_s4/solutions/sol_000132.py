# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state da2150ba) state=0e8873e4 sum of radii=1.695617 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Uses a hexagonal grid initialization and numerical optimization.
    """
    n = 26
    
    # --- 1. Initialization: Hexagonal Grid ---
    # We try to arrange 26 circles in a hexagonal pattern.
    # A common efficient shape is roughly square.
    # Let's try rows with counts: 5, 4, 5, 4, 5, 3 (Sum = 26)
    # Or 5, 5, 5, 5, 4, 2? 
    # Let's try to pack them tightly.
    
    # Heuristic: Place centers on a triangular lattice.
    # We can estimate a radius. For n=26, r ~ 0.1.
    # Grid spacing dx = 2r, dy = r*sqrt(3).
    
    # Let's generate points on a grid and pick 26 that fit best?
    # Or just define a specific pattern.
    # Pattern: 6 rows.
    # Row lengths: 5, 4, 5, 4, 5, 3 -> Total 26.
    # Row 0 (y=0): 5 circles. Shift 0.
    # Row 1 (y=1): 4 circles. Shift 0.5 (in units of 2r).
    # Row 2 (y=2): 5 circles. Shift 0.
    # Row 3 (y=3): 4 circles. Shift 0.5.
    # Row 4 (y=4): 5 circles. Shift 0.
    # Row 5 (y=5): 3 circles. Shift 0.5.
    
    # Let's estimate r initially as 0.1
    r_init = 0.1
    centers = []
    
    # Define row configurations (num_circles, shift_x_factor)
    # shift_x_factor: 0 for even rows, 0.5 for odd rows (in units of diameter)
    rows_config = [
        (5, 0.0),
        (4, 0.5),
        (5, 0.0),
        (4, 0.5),
        (5, 0.0),
        (3, 0.5)
    ]
    
    # Vertical spacing dy = r * sqrt(3)
    dy = r_init * math.sqrt(3)
    
    y = r_init
    for num_circles, shift in rows_config:
        # Horizontal spacing dx = 2r
        dx = 2 * r_init
        # We want to center the row in the square [0, 1]
        # Width occupied by row = (num_circles - 1) * dx
        # Start x should be (1 - width) / 2 + shift * dx? 
        # Actually, shift is relative to grid.
        # Let's just place them with some offset.
        
        # For a row of k circles, ideal span is [r, 1-r]
        # If k=5, 5 circles need width 4*2r = 8r? No.
        # Centers at r, r+2r, r+4r, r+6r, r+8r.
        # Last center r+8r. Right edge r+8r+r = 10r.
        # If r=0.1, 10r=1. Fits perfectly.
        # For k=4, centers r, r+2r, r+4r, r+6r. Right edge 8r.
        # We can center this row: start_x = r + (1 - 8r)/2 = r + 0.5 - 4r = 0.5 - 3r.
        # If r=0.1, start_x = 0.5 - 0.3 = 0.2.
        # Centers: 0.2, 0.4, 0.6, 0.8.
        
        # For k=3, width 4r. Start x = 0.5 - 2r.
        
        # Shift logic:
        # Even rows (0 shift) aligned with 5-circle grid?
        # Let's define a base grid for 5 circles: x = 0.1, 0.3, 0.5, 0.7, 0.9
        # 4 circles centered: x = 0.2, 0.4, 0.6, 0.8
        # 3 circles centered: x = 0.3, 0.5, 0.7
        
        # This creates a nice symmetric pattern.
        
        if num_circles == 5:
            x_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
        elif num_circles == 4:
            x_coords = [0.2, 0.4, 0.6, 0.8]
        elif num_circles == 3:
            x_coords = [0.3, 0.5, 0.7]
        else:
            # Fallback
            step = 1.0 / (num_circles + 1)
            x_coords = [i * step for i in range(1, num_circles + 1)]
            
        for x in x_coords:
            centers.append([x, y])
            
        y += dy

    centers = np.array(centers)
    
    # --- 2. Optimization ---
    # We want to maximize the minimum distance between any pair of centers
    # and between centers and boundaries.
    # Let this distance be D. Then max radius r = D/2.
    # We can maximize D.
    
    # However, standard optimizers minimize functions.
    # Objective: Maximize min( dist(i,j), dist(i, boundary) )
    # Let f(x) = - min_dist. Minimize f.
    
    # Variables: x1, y1, ..., x26, y26 (52 variables)
    
    def min_dist_sq(coords):
        # coords is flattened array of size 52
        pts = coords.reshape((n, 2))
        
        # Distance to boundaries squared
        # Boundary constraints: x >= r, x <= 1-r => dist to boundary >= r
        # If we work with diameter D = 2r, then dist to boundary >= D/2.
        # Let's optimize for D directly?
        # Actually, simpler to optimize for radius r directly.
        # But let's stick to maximizing the "clearance".
        
        # Let's calculate pairwise distances squared
        # and boundary distances squared.
        # We want to maximize the minimum of these values.
        # But scaling matters.
        
        # Let's just return the minimum distance between points and boundaries (as a radius limit).
        # If we find a configuration where min_dist = d, then we can set all r = d/2.
        
        min_d2 = 1e9
        
        # Point-to-point
        # Vectorized
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (26, 26, 2)
        dists_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists_sq, 1e9) # Ignore self
        min_pair_sq = np.min(dists_sq)
        
        # Point-to-boundary
        # x >= r, x <= 1-r => r <= x and r <= 1-x
        # y >= r, y <= 1-y => r <= y and r <= 1-y
        # The limiting radius for point i is min(x_i, 1-x_i, y_i, 1-y_i)
        # Let's compute this for all points.
        # But wait, if we assume equal radii, the global r is limited by the tightest point.
        # But here we are optimizing positions.
        # If we maximize the minimum distance between points (D_pair) and minimum distance to boundary (D_bound),
        # the max equal radius is min(D_pair/2, D_bound).
        
        # Let's compute min boundary distance
        x = pts[:, 0]
        y = pts[:, 1]
        dist_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        min_bound = np.min(dist_bound)
        
        # We want to maximize min(min_pair/2, min_bound)
        # Equivalent to maximizing min(min_pair, 2*min_bound) ?
        # No. r = min(min_pair/2, min_bound).
        # So we want to maximize r.
        # Let's just compute r_candidate = min(min_pair/2, min_bound)
        # And return -r_candidate.
        
        r_cand = min(np.sqrt(min_pair_sq)/2, min_bound)
        return -r_cand

    # Initial guess
    x0 = centers.flatten()
    
    # Bounds: centers in [0, 1]
    bounds = [(0, 1)] * 52
    
    # Use a robust optimizer. Nelder-Mead is good for non-smooth objectives (min function).
    # Or Powell.
    # Since the objective involves 'min', it's non-smooth. Nelder-Mead is safe.
    
    # However, Nelder-Mead might be slow or get stuck.
    # Let's try a few restarts or just run it.
    # 52 dimensions is large for Nelder-Mead? Maybe.
    # But let's try.
    
    res = minimize(min_dist_sq, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
    
    opt_centers = res.x.reshape((n, 2))
    opt_r = -res.fun
    
    # The optimization finds positions that maximize the minimum clearance.
    # We can set all radii to this optimal r.
    radii = np.full(n, opt_r)
    
    # --- 3. Verification and Potential Improvement ---
    # The above assumes equal radii.
    # Can we do better with unequal radii?
    # With equal radii r=opt_r, sum = 26 * opt_r.
    # If we allow unequal, some circles in "tight" spots might be smaller,
    # allowing others to grow?
    # But usually, the bottleneck is global.
    # Let's try a local refinement for unequal radii.
    # We can try to increase radii greedily or using a solver.
    
    # Let's formulate: Maximize sum(r_i)
    # Subject to: dist(i,j) >= r_i + r_j, boundary constraints.
    # This is hard.
    # But we can try a simple perturbation:
    # Fix centers, compute max valid radii for each.
    # r_i = min( x_i, 1-x_i, y_i, 1-y_i, min_{j!=i} (dist(i,j) - r_j) )
    # This is circular.
    # Iterative method:
    # Initialize r_i = opt_r (from equal packing).
    # While convergence:
    #   For each i:
    #     r_i = min( boundary_dist(i), min_{j!=i} (dist(i,j) - r_j) )
    #     Ensure r_i >= 0.
    # This is a form of relaxation.
    
    # However, this might decrease sum if not careful?
    # Actually, if we start with a valid packing, this computes the maximal radii for fixed centers.
    # Since our centers were optimized for equal radii, they are likely valid.
    # The computed radii will be >= opt_r (since opt_r was feasible).
    # Wait, if centers are fixed, the maximal radii are determined.
    # If the packing was tight (all constraints active), radii won't change much.
    # But if some circles are in looser spots, they can grow.
    
    # Let's compute the maximal radii for the optimized centers.
    # We can solve this as a linear program? No, dist is non-linear.
    # But we can iterate.
    
    current_radii = radii.copy()
    # Relaxation iterations
    for _ in range(50):
        new_radii = current_radii.copy()
        for i in range(n):
            # Distance to boundary
            x, y = opt_centers[i]
            r_bound = min(x, 1-x, y, 1-y)
            
            # Distance to others
            # r_i <= dist(i,j) - r_j  => r_i + r_j <= dist(i,j)
            # We want largest r_i satisfying this for all j.
            # r_i <= dist(i,j) - r_j
            # So r_i <= min_j (dist(i,j) - r_j)
            
            # We need to be careful with the order or simultaneous update.
            # Let's use a conservative update:
            # r_i_new = min(r_bound, min_{j!=i} (dist(i,j) - current_radii[j]))
            
            max_r = r_bound
            for j in range(n):
                if i == j: continue
                d = np.sqrt(np.sum((opt_centers[i] - opt_centers[j])**2))
                max_r = min(max_r, d - current_radii[j])
            
            # Ensure non-negative
            if max_r < 0:
                max_r = 0 # Should not happen if valid
            
            new_radii[i] = max_r
        
        # Check convergence
        if np.allclose(current_radii, new_radii, atol=1e-8):
            break
        current_radii = new_radii
        
    radii = current_radii
    
    # Final validation check (internal)
    # Ensure no overlaps with tolerance
    valid = True
    for i in range(n):
        x, y = opt_centers[i]
        r = radii[i]
        if r < 0: valid = False
        if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
            valid = False # Boundary violation
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((opt_centers[i] - opt_centers[j])**2))
            if d < radii[i] + radii[j] - 1e-9:
                valid = False # Overlap
                
    if not valid:
        # Fallback to equal radii if relaxation failed (shouldn't)
        radii = np.full(n, opt_r)

    sum_radii = np.sum(radii)
    return opt_centers, radii, sum_radii

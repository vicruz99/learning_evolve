# sol_000268 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c81ffe0) state=b087561b sum of radii=0.474253 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Returns (centers, radii, sum_radii) for 26 circles in a unit square.
    """
    n = 26
    
    # Initial guess: Hexagonal grid pattern
    # We try to place points in a hexagonal lattice and pick the first n
    # Then we optimize their positions to maximize the minimum distance (and thus radius)
    
    # Estimate radius to place initial points. 
    # If r=0.1, diameter=0.2. 5 circles fit in width.
    # Let's place them slightly smaller to ensure they fit initially
    r_init = 0.08
    
    # Generate hexagonal lattice points
    # Spacing x: 2*r, spacing y: sqrt(3)*r
    # But we don't know r yet. Let's use a generic grid density.
    # We want 26 points in [0,1]x[0,1].
    
    # Let's try a 6x5 grid or similar and cull?
    # Or just generate a dense grid and pick.
    
    # Better: Place centers on a grid, then optimize.
    # Let's create a grid of points that covers the square.
    # 6 columns, 5 rows = 30 points. We can pick 26.
    # But we want them distributed.
    
    # Let's try to place them in a way that mimics hexagonal packing.
    # 5 rows.
    # Row 0: 5 points
    # Row 1: 5 points
    # Row 2: 5 points
    # Row 3: 5 points
    # Row 4: 6 points? No, width constraint.
    # Maybe 5, 5, 5, 5, 5, 1?
    
    # Let's just generate a random set of points and run a repulsive force optimizer.
    # This is robust.
    
    # Initialization
    np.random.seed(42)
    centers = np.random.rand(n, 2)
    
    # To make it better, let's start with a grid
    # 6x5 grid -> 30 points. Remove 4 random?
    # Or 5x6 -> 30.
    # Let's do 5 rows, 6 cols?
    # x: 0.1, 0.3, 0.5, 0.7, 0.9 (5 cols)
    # y: 0.1, 0.3, 0.5, 0.7, 0.9 (5 rows) -> 25 points.
    # We need 26.
    # Add one at (0.5, 0.5)?
    
    # Let's create a structured initialization
    # 5 rows of 5, plus 1
    pts = []
    # 5 rows
    for r_idx in range(5):
        # Stagger rows
        if r_idx % 2 == 0:
            # 5 circles
            x_coords = np.linspace(0.1, 0.9, 5)
        else:
            # 5 circles, shifted
            # Shift by 0.1 (half spacing 0.2)
            x_coords = np.linspace(0.2, 0.8, 5) # 0.2, 0.4, 0.6, 0.8 -> 4 points?
            # We want 5. 0.2 to 0.8 step 0.2 gives 4 points (0.2, 0.4, 0.6, 0.8).
            # Maybe shift less.
            # Let's just use linspace with 5 points for all, centered.
            x_coords = np.linspace(0.1, 0.9, 5)
        
        y_coord = 0.1 + r_idx * 0.2 # 0.1, 0.3, 0.5, 0.7, 0.9
        for x in x_coords:
            pts.append([x, y_coord])
    
    # We have 25 points. Add 1.
    # Where? Center of a gap?
    # Maybe (0.5, 0.5) is already there?
    # Row 0 (even): 0.1, 0.3, 0.5, 0.7, 0.9 at y=0.1
    # Row 1 (odd): 0.1, 0.3, 0.5, 0.7, 0.9 at y=0.3 (not staggered in this simple init)
    # Let's make it staggered properly.
    
    pts = []
    # 5 rows
    # Row 0: 5 circles at y=0.1
    y0 = 0.1
    x0 = np.linspace(0.1, 0.9, 5)
    for x in x0:
        pts.append([x, y0])
        
    # Row 1: 5 circles at y=0.3, shifted by 0.1?
    # If shift 0.1, x range 0.2 to 1.0? No, 1.0 is boundary.
    # 0.2, 0.4, 0.6, 0.8. That's 4 circles.
    # Maybe we need 6 rows?
    
    # Let's try a different init: 6 rows of 4 or 5.
    # 4*6 = 24. Need 2 more.
    # 5*5 + 1 = 26.
    
    # Let's just use the repulsive force method which is robust to init.
    # Start with random points in (0.1, 0.9) to be safe.
    centers = np.random.uniform(0.1, 0.9, (n, 2))
    
    # Repulsive force optimization
    # We want to maximize the minimum distance between points and to walls.
    # This is equivalent to finding the largest r.
    
    # We will iteratively increase a target radius 'r_target' and push points apart.
    r_target = 0.05
    current_r = 0.05
    
    # Run for a number of steps
    # We can also use scipy to optimize the "min distance" directly.
    # Let's define a function that returns the minimum valid radius for a set of centers.
    
    def get_min_radius(centers):
        n = centers.shape[0]
        min_r = 1.0
        
        # Distance to walls
        # x >= r, x <= 1-r => r <= x, r <= 1-x
        # y >= r, y <= 1-y => r <= y, r <= 1-y
        dists_wall = np.minimum(
            np.minimum(centers[:, 0], 1 - centers[:, 0]),
            np.minimum(centers[:, 1], 1 - centers[:, 1])
        )
        min_r = min(min_r, np.min(dists_wall))
        
        # Distance between circles
        # 2r <= dist => r <= dist/2
        # Calculate pairwise distances
        # Use broadcasting
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
        dists = np.sqrt(np.sum(diff**2, axis=2)) # (n, n)
        np.fill_diagonal(dists, 1.0) # Ignore self
        min_dist = np.min(dists)
        min_r = min(min_r, min_dist / 2.0)
        
        return min_r

    # We want to maximize this function.
    # Let's use scipy.optimize to maximize get_min_radius.
    # But get_min_radius is not smooth.
    # However, it's piecewise smooth.
    # We can use a derivative-free optimizer like Nelder-Mead or Powell.
    # Or we can use a smooth approximation.
    # A common smooth approximation for min(x1, x2, ...) is -log(sum(exp(-x))) / alpha or similar.
    # Or just use Powell which handles non-smooth functions reasonably well.
    
    # To improve chances, run multiple times with different seeds.
    
    best_centers = None
    best_r = 0.0
    
    # We can parameterize centers as a flat array
    def objective(x_flat):
        c = x_flat.reshape((n, 2))
        # Clamp to valid range slightly inside to avoid boundary issues in optimization
        # But the function handles boundary penalties? No, it calculates radius.
        # If center is outside [0,1], radius might be negative?
        # We should keep centers inside [0,1].
        # Optimization bounds will handle this.
        return -get_min_radius(c) # Minimize negative radius

    # Bounds: [0, 1] for all coords
    bounds = [(0, 1)] * (2 * n)
    
    # Let's try a few restarts
    for seed in range(5):
        np.random.seed(seed)
        x0 = np.random.uniform(0.1, 0.9, (n, 2)).flatten()
        
        # Use Powell method
        res = opt.minimize(objective, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-8})
        
        if res.success or res.fun < -best_r:
            r_val = -res.fun
            if r_val > best_r:
                best_r = r_val
                best_centers = res.x.reshape((n, 2))
                
        # Also try L-BFGS-B with a smooth approximation?
        # Let's try a smooth approximation: sum of 1/dist? No.
        # Soft min: - (1/kappa) * log(sum(exp(-kappa * val)))
        # val = dist/2 or dist_wall
        # We want to maximize min(val).
        # Approx: -log(mean(exp(-kappa * val))) / kappa ?
        # Actually, if we want to maximize min(x_i), we can minimize sum(exp(-x_i)).
        # Or use the "log-sum-exp" trick for min.
        # min(x) approx - (1/a) log(sum(exp(-a*x)))
        # So we want to minimize - (-1/a log(...)) = 1/a log(sum(exp(-a*x)))
        # Wait, max min(x) <=> min -min(x) <=> min max(-x)
        # -min(x_i) = max(-x_i)
        # Softmax approximation for max(-x_i) is (1/a) log(sum(exp(a*(-x_i))))
        # So we minimize (1/a) log(sum(exp(-a*x_i)))
        
        # Let's implement this smooth objective
        def smooth_objective(x_flat):
            c = x_flat.reshape((n, 2))
            # Wall distances
            dw = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), 
                            np.minimum(c[:, 1], 1 - c[:, 1]))
            
            # Pairwise distances / 2
            # To avoid computing full matrix every time, maybe just sample?
            # But for 26, it's fast.
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, 1e9) # Large value to ignore
            r_pairs = dists / 2.0
            
            # Combine all constraints into one list
            # r_candidates = np.concatenate([dw, r_pairs.reshape(-1)])
            # But r_pairs has n^2 elements.
            # Let's just use dw and some random pairs? No, all pairs matter.
            
            # Actually, the bottleneck is likely a few pairs.
            # But to be safe, let's use all.
            # Vectorize.
            
            # r_vals = min(dw) and min(r_pairs)
            # We want max(min(dw, min(r_pairs)))
            
            # Let's construct the array of all constraints
            # This might be memory heavy for large n, but n=26 is tiny.
            r_all = np.concatenate([dw, r_pairs.flatten()])
            
            # Smooth min approximation
            # max_radius = min(r_all)
            # We want to maximize this.
            # Use log-sum-exp for min.
            # min(x) approx -1/a * log(sum(exp(-a*x)))
            # We want to maximize this, so minimize -min(x) approx 1/a * log(sum(exp(-a*x)))
            
            a = 500.0 # Sharpness
            # Avoid overflow
            # r_all should be positive.
            # If r_all is very small, exp(-a*r) is large? No, if r is small, -a*r is small negative (close to 0), exp is 1.
            # If r is large, -a*r is large negative, exp is 0.
            # So sum(exp(-a*r)) is dominated by small r.
            # Wait. We want min(r). The smallest r gives the largest exp(-a*r)?
            # exp(-a*small) is close to 1. exp(-a*large) is close to 0.
            # Yes. So sum is dominated by the smallest elements.
            # log(sum) will be roughly log(count_small).
            # 1/a * log(...) will be small.
            # We want to minimize this value.
            # As min(r) increases, exp(-a*r) decreases for all.
            # So sum decreases. log decreases. Value decreases.
            # So minimizing this approximates maximizing min(r).
            
            # But we need to handle the case where r_all can be 0 or negative (if overlapping).
            # If r_all < 0, exp(-a*r) explodes.
            # So this penalty heavily penalizes negative radii (overlaps).
            
            # Clamp r_all to be at least some small positive?
            # Or just let it explode.
            
            # To prevent overflow, cap exponent
            exponents = -a * r_all
            # Clip exponents to avoid overflow
            exponents = np.clip(exponents, -500, 500)
            
            log_sum_exp = np.log(np.sum(np.exp(exponents)))
            val = log_sum_exp / a
            
            return val

        # Try optimizing the smooth version
        try:
            res2 = opt.minimize(smooth_objective, x0, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 500})
            r_val2 = -get_min_radius(res2.x.reshape((n, 2))) # Check actual min radius
            if r_val2 > best_r:
                best_r = r_val2
                best_centers = res2.x.reshape((n, 2))
        except:
            pass

    # After optimization, we have best_centers and best_r.
    # However, get_min_radius gives the maximum EQUAL radius.
    # The problem asks to maximize SUM of radii.
    # With equal radii, sum = 26 * best_r.
    # Is it possible to do better with unequal radii?
    # Maybe. But usually equal is very close to optimal.
    # Let's check if we can improve by allowing unequal radii.
    # But optimizing 26 radii + 52 coords is hard.
    # Let's stick to equal radii as a strong baseline.
    # If best_r * 26 >= 2.636, we are good.
    
    # Let's check the value.
    # If best_r is around 0.1014, sum is 2.636.
    # Let's hope the optimizer finds it.
    
    # One final check: ensure no overlaps with the calculated radius.
    # The optimizer might return a configuration where min_radius is slightly violated due to numerical errors or smooth approx.
    # We should compute the exact radius for the best_centers.
    
    final_r = get_min_radius(best_centers)
    
    # If the radius is very small or invalid, fallback?
    # But it should be fine.
    
    # Just to be safe, clamp radii to be non-negative
    final_r = max(final_r, 0.0)
    
    radii = np.full(n, final_r)
    
    # Validate
    # (We can't run the validation function here, but we trust the logic)
    
    # One detail: The optimization might push centers outside [0,1] if bounds are not respected strictly?
    # L-BFGS-B respects bounds. Powell does not.
    # If we used Powell, we might have centers outside.
    # Let's clamp centers to [0, 1] just in case, and recompute radius.
    # But if we clamp, we might introduce overlaps?
    # Actually, if center is outside, the distance to wall is negative (or handled by logic).
    # Our get_min_radius handles walls correctly (returns negative if outside).
    # So if best_r > 0, centers must be inside (mostly).
    # But numerical errors might place them at 1.0000001.
    # Let's clamp.
    best_centers = np.clip(best_centers, 0, 1)
    final_r = get_min_radius(best_centers)
    radii = np.full(n, final_r)
    
    return best_centers, radii, np.sum(radii)

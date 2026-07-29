# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b0810f40) state=78f93b39 sum of radii=2.210000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns centers, radii, and sum_radii.
    """
    n = 26
    np.random.seed(42) # For reproducibility

    # Function to compute the overlap penalty for a given configuration and radius
    # Minimizing this function pushes circles apart
    def compute_penalty(centers, R):
        penalty = 0.0
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < 2 * R:
                    penalty += (2 * R - dist) ** 2
        return penalty

    # Function to attempt packing for a fixed radius R
    # Returns (success, centers, min_penalty)
    def try_pack_radius(R, restarts=3):
        best_penalty = float('inf')
        best_centers = None
        
        # Bounds for centers: [R, 1-R] for x and y
        bounds = [(R, 1 - R) for _ in range(2 * n)]
        
        # We will try multiple initial configurations
        # 1. Random initialization
        # 2. Hexagonal grid initialization
        
        inits = []
        
        # Random initialization
        inits.append(np.random.uniform(R, 1 - R, size=(n, 2)))
        
        # Hexagonal grid initialization
        # Try to fit points in a hex pattern
        hex_centers = []
        r_approx = R
        # Rows
        rows = 6
        cols = 5
        y = r_approx
        for r_idx in range(rows):
            x = r_approx
            # Offset every other row
            offset = (r_idx % 2) * r_approx
            current_x = x + offset
            while current_x + r_approx <= 1.0 and len(hex_centers) < n:
                hex_centers.append([current_x, y])
                current_x += 2 * r_approx
            y += np.sqrt(3) * r_approx
        
        if len(hex_centers) >= n:
            # Take first n
            inits.append(np.array(hex_centers[:n]))
        else:
            # If hex grid didn't fill, pad with random
            # (Unlikely given R ~ 0.1)
            pass

        # Add a few more random ones for robustness
        for _ in range(restarts - 2): # -2 because we added 2 specific inits
             inits.append(np.random.uniform(R, 1 - R, size=(n, 2)))

        for centers_init in inits:
            x0 = centers_init.flatten()
            
            # Define objective for scipy
            def obj(x):
                c = x.reshape(n, 2)
                # Boundary penalties (soft, though bounds are hard)
                # Hard bounds are enforced by optimizer, but soft penalty helps gradient
                # Actually L-BFGS-B handles bounds well, we just need overlap penalty
                return compute_penalty(c, R)

            try:
                res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                               options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 2000})
                
                if res.fun < best_penalty:
                    best_penalty = res.fun
                    best_centers = res.x.reshape(n, 2)
                    
                    # Early exit if very good
                    if best_penalty < 1e-8:
                        return True, best_centers, best_penalty
            except:
                continue
                
        # Check if we found a valid packing
        # Tolerance for overlap sum. With N=26, small overlaps sum up.
        # We want max overlap to be small.
        # A safe heuristic: if penalty is very small, it's likely valid.
        # But let's check max overlap explicitly.
        if best_centers is not None:
            max_overlap = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                    if d < 2 * R:
                        ov = 2 * R - d
                        if ov > max_overlap:
                            max_overlap = ov
            
            # If max overlap is negligible, success
            if max_overlap < 1e-7:
                return True, best_centers, best_penalty
        
        return False, best_centers, best_penalty

    # Binary search for optimal R
    low = 0.05
    high = 0.12 # Theoretical limit is ~0.105, 0.12 is safe upper bound
    optimal_R = low
    optimal_centers = None
    valid_found = False

    # We can also try to expand high if low succeeds easily, but 0.12 is likely enough.
    # Let's do a coarser search first to find range, then refine.
    
    # Step 1: Find a feasible R
    # Check mid
    mid = (low + high) / 2
    # Check if high is feasible?
    # If high is feasible, we can go higher. But 0.12 is probably max.
    
    # Binary search iterations
    for _ in range(25):
        if high - low < 1e-5:
            break
            
        mid = (low + high) / 2
        
        # Try to pack with radius mid
        # Use fewer restarts for speed, or more if stuck?
        # 3 restarts is a good balance
        success, centers, penalty = try_pack_radius(mid, restarts=3)
        
        if success:
            optimal_R = mid
            optimal_centers = centers
            low = mid
            valid_found = True
            # Try slightly larger
            high = mid + 1e-5 
        else:
            high = mid

    # If we didn't find a valid packing in binary search (unlikely), 
    # fallback to a known good configuration.
    # But with the loop, we should find something.
    
    if not valid_found or optimal_centers is None:
        # Fallback: Simple grid
        # 5x5 grid with r=0.1
        R_fallback = 0.1
        centers = []
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + i*0.2, 0.1 + j*0.2])
        # We have 25 circles, need 26.
        # Add one in center? Center is (0.5, 0.5) which is occupied.
        # Add at (0.5, 0.5) with smaller radius? 
        # But we want equal radii.
        # Just randomize 26th
        centers.append([0.2, 0.2]) # Overlaps, but fallback
        # This fallback is weak.
        # Better to rely on optimization.
        pass

    # Final Refinement
    # We have a configuration with radius optimal_R that is nearly valid.
    # We can try to slightly decrease radius to ensure strict validity if needed,
    # or just trust the optimizer.
    # To be safe, let's run one more optimization step with the found centers 
    # and current R to ensure minimal energy.
    
    if valid_found:
        # Re-optimize with found centers as seed to polish
        R_final = optimal_R
        # Tighten R slightly if there were any tiny overlaps
        # But try_pack returned success only if max_overlap < 1e-7
        
        # Let's verify and adjust R if necessary
        # Check constraints
        current_centers = optimal_centers
        # Check boundaries
        # Check overlaps
        max_ov = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((current_centers[i] - current_centers[j]) ** 2))
                if d < 2 * R_final:
                    max_ov = max(max_ov, 2 * R_final - d)
        
        # If there is a tiny overlap, reduce R
        if max_ov > 1e-9:
            R_final = R_final - max_ov - 1e-10
            # Re-center? No, just shrinking R fixes overlap relative to centers.
            # But might cause boundary issues if centers are too close to edge.
            # However, centers were optimized with bounds [R, 1-R] for the larger R.
            # So for smaller R, they are strictly inside.
            
        # Create radii array
        radii = np.full(n, R_final)
        
        # One last check: ensure centers are within [R_final, 1-R_final]
        # Since we optimized with bounds [optimal_R, 1-optimal_R] and optimal_R >= R_final,
        # centers are valid.
        
        sum_radii = np.sum(radii)
        
        return current_centers, radii, sum_radii

    # Fallback return (should not be reached if logic is correct)
    return np.random.rand(26, 2) * 0.8 + 0.1, np.ones(26) * 0.05, 1.3

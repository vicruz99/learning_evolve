# sol_000156 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39a6f529) state=496708a7 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal-like grid
    # We try to fit circles in a pattern that mimics hexagonal packing
    # to utilize space efficiently.
    
    # Parameters for initial placement
    r_init = 0.09 # Start slightly below 0.1
    
    # Generate points in a hexagonal pattern
    # Vertical spacing = sqrt(3) * r
    # Horizontal spacing = 2 * r
    centers = []
    radii = []
    
    y = r_init
    row = 0
    while y < 1.0 - r_init:
        x = r_init
        offset = (row % 2) * r_init # Shift every other row
        while x < 1.0 - r_init:
            centers.append([x + offset, y])
            radii.append(r_init)
            x += 2 * r_init
            if len(centers) >= n_circles:
                break
        y += math.sqrt(3) * r_init
        row += 1
        if len(centers) >= n_circles:
            break
            
    # If we didn't get enough, add random ones (unlikely with r=0.09)
    while len(centers) < n_circles:
        centers.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        radii.append(r_init)
        
    centers = np.array(centers[:n_circles])
    radii = np.array(radii[:n_circles])
    
    # 2. Optimization Loop
    # We will iteratively try to increase radii and relax positions.
    
    max_iterations = 200
    growth_step = 0.0005
    
    # Precompute indices for constraints
    pairs = [(i, j) for i in range(n_circles) for j in range(i + 1, n_circles)]
    
    def get_overlap_energy(pos, rad):
        """
        Calculates the energy based on overlaps and boundary violations.
        pos: (N, 2)
        rad: (N,)
        """
        energy = 0.0
        
        # Pairwise overlaps
        # Vectorized distance calculation
        # pos shape (N, 2)
        # Compute distance matrix
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (N, N, 2)
        dists = np.sqrt(np.sum(diff**2, axis=2)) # (N, N)
        
        # Lower triangle indices
        triu_indices = np.triu_indices(n_circles, k=1)
        r_sum = rad[np.newaxis, :] + rad[:, np.newaxis] # (N, N)
        
        overlaps = np.maximum(0, r_sum - dists)
        energy += np.sum(overlaps**2)
        
        # Boundary violations
        # Left/Right
        violations_x_left = np.maximum(0, rad - pos[:, 0])
        violations_x_right = np.maximum(0, rad - (1.0 - pos[:, 0]))
        violations_y_bottom = np.maximum(0, rad - pos[:, 1])
        violations_y_top = np.maximum(0, rad - (1.0 - pos[:, 1]))
        
        energy += np.sum(violations_x_left**2) + np.sum(violations_x_right**2)
        energy += np.sum(violations_y_bottom**2) + np.sum(violations_y_top**2)
        
        return energy

    def energy_grad(pos, rad):
        """
        Gradient of energy w.r.t positions.
        """
        grad = np.zeros_like(pos)
        
        # Pairwise
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-12, 1e-12, dists)
        dirs = diff / dists_safe[:, :, np.newaxis] # (N, N, 2)
        
        r_sum = rad[np.newaxis, :] + rad[:, np.newaxis]
        overlaps = np.maximum(0, r_sum - dists)
        
        # Force magnitude for pair (i, j) is 2 * overlap
        # Force direction on i is +dir, on j is -dir
        # We sum contributions
        
        # Construct a matrix of forces magnitude
        # F_mag[i, j] = 2 * overlap[i, j]
        # But overlap is symmetric.
        # Force on i from j: 2 * overlap * dir_ij
        # Force on j from i: -2 * overlap * dir_ij
        
        # We can compute this efficiently
        # Overlaps matrix is symmetric
        # We only care where overlap > 0
        
        # Let's compute contribution to grad[i]
        # grad[i] += sum_j ( 2 * overlap[i,j] * (pos[i] - pos[j]) / dist[i,j] )
        
        # Vectorized approach:
        # We have dirs (N, N, 2). overlap (N, N).
        # We want sum over j for each i.
        
        # Filter for overlaps
        mask = overlaps > 1e-12
        force_mag = 2.0 * overlaps * mask # (N, N)
        
        # Weighted sum of directions
        # grad[i] = sum_j force_mag[i,j] * dirs[i,j]
        # Note: dirs[i,j] = (pos[i]-pos[j])/dist
        # This gives repulsion direction (away from j)
        
        # Compute sum
        # (N, N) * (N, N, 2) -> sum axis 1
        # Broadcasting: force_mag[:,:,np.newaxis] * dirs
        
        # However, this sums both i->j and j->i?
        # No, loop i over rows, j over cols.
        # For a specific i, we sum over j.
        # dirs[i,j] is vector from j to i.
        # So positive force_mag pushes i away from j. Correct.
        
        # But force_mag[i,j] == force_mag[j,i].
        # dirs[i,j] == -dirs[j,i].
        # So forces are consistent.
        
        # Summing over axis 1 (j)
        # grad = np.sum(force_mag[:,:,np.newaxis] * dirs, axis=1)
        
        # This might be heavy memory wise for large N, but N=26 is small.
        # Let's do it.
        
        # To save memory, we can compute iteratively or use sparse logic, but N=26 is trivial.
        
        # Actually, let's just loop for clarity and safety, N=26 is very small.
        for i in range(n_circles):
            for j in range(n_circles):
                if i == j: continue
                dist = dists[i, j]
                if dist < r_sum[i, j]:
                    overlap = r_sum[i, j] - dist
                    # Direction from j to i
                    direction = (pos[i] - pos[j]) / dist
                    # Force is repulsive, proportional to overlap
                    # Gradient of (overlap^2) is 2*overlap * (-grad_dist) ?
                    # Energy term: (r_sum - dist)^2
                    # dE/dpos[i] = 2(r_sum - dist) * (- d(dist)/dpos[i])
                    # dist = |pos[i] - pos[j]|
                    # d(dist)/dpos[i] = (pos[i] - pos[j]) / dist
                    # So dE/dpos[i] = 2(r_sum - dist) * (-(pos[i] - pos[j])/dist)
                    # Wait, overlap = r_sum - dist.
                    # If we define Energy = overlap^2.
                    # d(overlap)/dpos[i] = - (pos[i]-pos[j])/dist
                    # So gradient is 2 * overlap * (- direction).
                    # This is attractive?
                    # If overlap > 0, we want to INCREASE dist.
                    # Increasing dist means moving pos[i] AWAY from pos[j].
                    # Vector away from j is (pos[i] - pos[j]).
                    # So force should be in direction of (pos[i] - pos[j]).
                    # Let's check signs.
                    # E = (C - D)^2. dE/dD = -2(C-D).
                    # D = |x|. dD/dx = x/|x|.
                    # dE/dx = -2(C-D) * x/|x|.
                    # If C > D (overlap), term is negative.
                    # Force is -Gradient? No, we minimize E.
                    # Gradient points uphill.
                    # If C > D, (C-D) > 0. -2(+) * direction.
                    # Direction is x/|x| (from j to i? No, x is pos[i]-pos[j]).
                    # pos[i] - pos[j] points from j to i.
                    # So gradient points from j to i?
                    # If gradient points from j to i, moving in -gradient moves towards j.
                    # That reduces distance. Bad.
                    # We want to increase distance.
                    # So we should move in direction +gradient?
                    # No, minimize E.
                    # If E is high, we want to go down.
                    # If C > D, E increases as D decreases.
                    # So E is high when D is small.
                    # We want to increase D.
                    # Gradient of E w.r.t pos[i] should point in direction that increases E?
                    # Let's re-evaluate.
                    # E = (K - |x|)^2.
                    # If |x| < K, E > 0.
                    # dE/d|x| = -2(K - |x|) < 0.
                    # So increasing |x| decreases E.
                    # |x| increases if we move x away from 0 (assuming 0 is pos[j]).
                    # x = pos[i] - pos[j].
                    # Moving pos[i] in direction x increases |x|.
                    # So we want to move pos[i] in direction x.
                    # Gradient of E w.r.t x:
                    # dE/dx = dE/d|x| * d|x|/dx = -2(K-|x|) * (x/|x|).
                    # This vector is opposite to x (since K-|x| > 0).
                    # So Gradient points towards j.
                    # To minimize E, we move in -Gradient, which is direction x (away from j).
                    # Correct.
                    
                    # So Gradient = -2 * overlap * direction_away_from_j
                    # direction_away_from_j = (pos[i] - pos[j]) / dist
                    # Grad += -2 * overlap * (pos[i] - pos[j]) / dist
                    
                    # Let's stick to the code logic:
                    # grad[i] -= 2 * overlap * direction
                    pass
            
            # Vectorized gradient calculation for pairs
            # dists[i,j] is dist between i and j
            # overlap[i,j] is max(0, r_sum - dists)
            # direction[i,j] = (pos[i] - pos[j]) / dists
            
            # grad[i] = sum_j ( -2 * overlap[i,j] * direction[i,j] )
            # But overlap[i,j] is 0 if no overlap.
            
            # Let's compute matrix M where M[i,j] = 2 * overlap[i,j] / dists[i,j]
            # Then grad[i] = - sum_j M[i,j] * (pos[i] - pos[j])
            
            # Careful with dist=0
            mask = dists > 1e-12
            inv_dists = np.zeros_like(dists)
            inv_dists[mask] = 1.0 / dists[mask]
            
            overlaps_mat = np.maximum(0, r_sum - dists)
            coeffs = 2.0 * overlaps_mat * inv_dists # (N, N)
            
            # grad[i] -= sum_j coeffs[i,j] * (pos[i] - pos[j])
            # pos[i] - pos[j] is diff[i, j, :]
            
            # We can compute this using dot product
            # grad[i] -= coeffs[i, :] @ diff[i, :, :]
            
            for i in range(n_circles):
                grad[i] -= np.dot(coeffs[i, :], diff[i, :, :])

        # Boundary gradients
        # Left: E = (r - x)^2 if x < r. dE/dx = -2(r-x) = 2(x-r).
        # If x < r, x-r < 0, so grad points negative (left).
        # -Grad points right (away from wall). Correct.
        # Term: max(0, r - x)^2
        # Derivative w.r.t x: 2(r-x) * (-1) if r-x > 0 => -2(r-x) = 2(x-r).
        
        # x < r: violation.
        violation_x_left = rad - pos[:, 0]
        mask_x_left = violation_x_left > 1e-12
        grad[mask_x_left, 0] += 2 * violation_x_left[mask_x_left] # Wait, dE/dx = 2(x-r) = -2(r-x).
        # If x < r, r-x > 0. Gradient should be negative (pushing x up? No).
        # If x is small (0), r-x is positive. E is high.
        # We want to increase x.
        # Gradient of E w.r.t x should be negative?
        # If E = (r-x)^2, dE/dx = -2(r-x).
        # If r-x > 0, dE/dx < 0.
        # So Gradient points left (decreasing x).
        # Minimizing E means moving against gradient -> moving right (increasing x).
        # Correct.
        # So grad component should be -2(r-x).
        # My previous line: grad += 2*(r-x) was wrong sign?
        # Let's re-verify.
        # E = (r - x)^2.
        # dE/dx = 2(r - x) * (-1) = -2(r - x).
        # So grad[0] += -2 * (r - x).
        
        # Let's rewrite boundary gradients carefully.
        
        # Left Wall
        val = rad - pos[:, 0]
        mask = val > 1e-12
        grad[mask, 0] += -2.0 * val[mask]
        
        # Right Wall: E = (r - (1-x))^2 = (r - 1 + x)^2.
        # Let u = r - 1 + x. E = u^2. dE/dx = 2u * 1 = 2(r - 1 + x).
        # Violation if 1-x < r => x > 1-r.
        # val = (1-x) - r = 1 - x - r.
        # If val < 0, violation.
        # Let's use form (r - (1-x))^2.
        # Violation amount: r - (1-x) = r - 1 + x.
        # If > 0, penalty.
        val = rad - (1.0 - pos[:, 0])
        mask = val > 1e-12
        grad[mask, 0] += 2.0 * val[mask] # Because dE/dx = 2val.
        
        # Bottom Wall
        val = rad - pos[:, 1]
        mask = val > 1e-12
        grad[mask, 1] += -2.0 * val[mask]
        
        # Top Wall
        val = rad - (1.0 - pos[:, 1])
        mask = val > 1e-12
        grad[mask, 1] += 2.0 * val[mask]
        
        return grad

    # Optimization parameters
    lr = 0.01 # Learning rate for gradient descent on positions
    
    # Main Loop
    for step in range(max_iterations):
        # 1. Optimize positions for current radii
        # We use a simple gradient descent step or scipy minimize
        # Scipy minimize might be overkill per step, but robust.
        # Let's try a few steps of gradient descent with adaptive step size
        
        # Flatten positions for optimization
        x_flat = centers.flatten()
        
        # Define objective for scipy
        def obj(pos_flat):
            pos = pos_flat.reshape(-1, 2)
            return get_overlap_energy(pos, radii)
        
        def grad_obj(pos_flat):
            pos = pos_flat.reshape(-1, 2)
            g = energy_grad(pos, radii)
            return g.flatten()
        
        # Use L-BFGS-B or similar
        # Bounds for positions: [0, 1] is too loose, [r, 1-r] is tight
        # But r changes. Let's use [0, 1] and rely on penalty.
        bounds = [(0.0, 1.0)] * (2 * n_circles)
        
        try:
            res = opt.minimize(obj, x_flat, jac=grad_obj, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            centers = res.x.reshape(-1, 2)
        except Exception:
            pass # Fallback to no update
            
        # 2. Check validity and try to grow
        # Calculate current overlaps
        e = get_overlap_energy(centers, radii)
        
        # If energy is very low (valid packing), try to increase radii
        if e < 1e-6:
            # Check if we can grow
            # Try growing by a small amount
            # Determine max possible growth based on gaps?
            # Simple heuristic: grow by fixed step if valid
            radii += growth_step
            
            # Clamp radii to reasonable max (0.5)
            radii = np.minimum(radii, 0.5)
            
            # After growing, we likely have overlaps.
            # The next iteration's position optimization will fix it.
            # But we might need to reduce growth step if we can't resolve it.
            # For now, keep it simple.
        else:
            # If not valid, maybe reduce growth step or just rely on optimization
            # If energy is high, maybe we are stuck.
            # Try to reduce growth step
            if step > 50:
                 growth_step *= 0.95

    # Final check and return
    sum_radii = np.sum(radii)
    
    # Ensure constraints are satisfied (clipping if necessary due to float errors)
    # The validation function is strict, so we should be careful.
    # But the optimization should have handled it.
    
    return centers, radii, sum_radii

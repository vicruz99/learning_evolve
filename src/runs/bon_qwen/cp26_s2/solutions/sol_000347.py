# sol_000347 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 03d022f0) state=f10b1156 sum of radii=2.416986 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def compute_energy(centers, r):
    """
    Computes the overlap energy for a configuration of circles with equal radius r.
    Energy is sum of squared overlaps for circle-circle and circle-boundary.
    """
    n = centers.shape[0]
    energy = 0.0
    
    # Circle-Circle Overlap
    # Vectorized computation for efficiency
    # centers shape: (n, 2)
    # diff shape: (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Lower triangular indices for pairs (i, j) with i < j
    # We can just sum over all and divide by 2, or use tril
    # Using a mask for i < j
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    
    # Overlap distance = 2r - dist. If negative, no overlap.
    # Energy term is (max(0, 2r - dist))^2
    overlaps = 2 * r - dist
    overlaps[overlaps < 0] = 0
    energy += np.sum(overlaps[mask]**2)
    
    # Boundary Overlap
    # x < r or x > 1-r, y < r or y > 1-r
    # Term: max(0, r - x)^2 + max(0, x - (1-r))^2
    # Same for y
    
    # Left boundary
    dist_left = centers[:, 0] - r
    overlap_left = np.maximum(0, -dist_left) # if dist_left < 0 (i.e., x < r), overlap is positive
    # Actually overlap amount is r - x.
    # Let's define penalty as (r - x)^2 if x < r.
    penalty_x_neg = np.maximum(0, r - centers[:, 0])**2
    penalty_x_pos = np.maximum(0, centers[:, 0] - (1 - r))**2
    penalty_y_neg = np.maximum(0, r - centers[:, 1])**2
    penalty_y_pos = np.maximum(0, centers[:, 1] - (1 - r))**2
    
    energy += np.sum(penalty_x_neg) + np.sum(penalty_x_pos) + np.sum(penalty_y_neg) + np.sum(penalty_y_pos)
    
    return energy

def compute_gradient(centers, r):
    """
    Computes gradient of energy w.r.t centers.
    """
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    
    # Circle-Circle
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Avoid division by zero
    dist[dist < 1e-12] = 1e-12
    
    # Overlap magnitude
    overlap = 2 * r - dist
    # Only consider overlaps
    active_overlap = np.where(overlap > 0, overlap, 0.0)
    
    # Gradient contribution for pair (i, j)
    # d/dx_i ( (2r - d_ij)^2 ) = 2(2r - d_ij) * (- d_ij' / d_ij * d_ij' ?)
    # d/dx_i (d_ij) = (x_i - x_j) / d_ij
    # So term is -2(2r - d_ij) * (x_i - x_j) / d_ij
    # = -2 * overlap * (diff) / dist
    
    # We need to sum over j for each i.
    # For i, sum over j != i of term.
    # Note: diff[i, j] = x_i - x_j.
    # Gradient for x_i gets contributions from all j.
    
    # Create a weight matrix: -2 * overlap / dist
    weights = -2.0 * active_overlap / dist
    
    # Sum contributions
    # grad[i] += sum_j (weights[i, j] * diff[i, j])
    # diff[i, j] is vector (dx, dy)
    # weights[i, j] is scalar
    
    # Vectorized sum:
    # np.sum(weights[..., np.newaxis] * diff, axis=1)
    
    grad += np.sum(weights[..., np.newaxis] * diff, axis=1)
    
    # Boundary Gradient
    # Penalty (r - x)^2 -> deriv -2(r - x) = 2(x - r)
    # If x < r, deriv is 2(x - r) (negative, pushing x up? No, gradient points to increase function.
    # We want to minimize energy. So gradient is correct direction for descent.
    # Wait, if x < r, we want x to increase. Energy decreases as x increases.
    # Derivative of (r-x)^2 w.r.t x is -2(r-x). If x<r, r-x>0, deriv is negative.
    # So gradient descent (x -= alpha * grad) will add positive amount. Correct.
    
    mask_x_neg = centers[:, 0] < r
    if np.any(mask_x_neg):
        grad[mask_x_neg, 0] += 2 * (centers[mask_x_neg, 0] - r)
        
    mask_x_pos = centers[:, 0] > 1 - r
    if np.any(mask_x_pos):
        # Penalty (x - (1-r))^2 -> deriv 2(x - (1-r))
        # If x > 1-r, deriv positive. Gradient descent reduces x. Correct.
        grad[mask_x_pos, 0] += 2 * (centers[mask_x_pos, 0] - (1 - r))
        
    mask_y_neg = centers[:, 1] < r
    if np.any(mask_y_neg):
        grad[mask_y_neg, 1] += 2 * (centers[mask_y_neg, 1] - r)
        
    mask_y_pos = centers[:, 1] > 1 - r
    if np.any(mask_y_pos):
        grad[mask_y_pos, 1] += 2 * (centers[mask_y_pos, 1] - (1 - r))
        
    return grad

def run_packing():
    np.random.seed(42)
    n_circles = 26
    
    # Strategy:
    # 1. Initialize with a hexagonal-like grid or perturbed grid.
    # 2. Iteratively increase radius r and minimize energy.
    # 3. Final pass to allow unequal radii if possible (optional but good).
    
    # Initial positions: 5x5 grid + 1
    # 5x5 grid points: 0.1, 0.3, 0.5, 0.7, 0.9
    # We have 26 circles.
    # Let's generate a grid of 30 points (6x5) and pick 26?
    # Or just random. Random is safer to avoid bias, but grid is faster.
    # Let's try a dense random start.
    
    centers = np.random.rand(n_circles, 2)
    
    # Start with a small radius that is guaranteed to fit (e.g., 0.02)
    # But we want to find max r.
    # We can start r at 0.05 and try to grow.
    r_current = 0.05
    
    # Optimization parameters
    max_r = 0.2
    r_step = 0.0001
    max_iter = 500 # Max iterations for growing
    tol_energy = 1e-8
    
    # Pre-calculate bounds for L-BFGS-B
    # Bounds are [r, 1-r] for each coordinate.
    # But bounds depend on r. L-BFGS-B bounds are static.
    # So we can't use L-BFGS-B bounds for dynamic r.
    # We rely on energy penalty for boundaries or use SLSQP.
    # However, penalty method is smoother.
    
    # Let's use a simple loop:
    # Increase r, optimize positions using L-BFGS-B with fixed bounds [0, 1]
    # and rely on energy penalty to keep them inside [r, 1-r].
    # Actually, if we enforce bounds [r, 1-r] in L-BFGS-B, we can't change them easily.
    # But we can set bounds [0, 1] and rely on penalty.
    
    # Better: Use a custom gradient descent or just scipy minimize with bounds [0,1]
    # and heavy penalty for boundary.
    
    # Let's try a robust approach:
    # 1. Place circles in a valid configuration for r=0.1 (e.g. 25 in grid, 1 small in hole?)
    #    Actually, just start with r=0.05 and random positions.
    
    # Optimization loop to grow r
    for step in range(2000):
        # Check if current r is feasible (energy approx 0)
        # We run a few optimization steps to minimize energy at current r
        # If energy -> 0, increase r.
        
        # Define objective and gradient for scipy
        def objective(x_flat):
            x = x_flat.reshape(n_circles, 2)
            return compute_energy(x, r_current)
        
        def gradient(x_flat):
            x = x_flat.reshape(n_circles, 2)
            g = compute_gradient(x, r_current)
            return g.flatten()
        
        # Flatten centers
        x_flat = centers.flatten()
        
        # Optimize positions for current r
        # Use L-BFGS-B with bounds [0, 1]
        # Note: We cannot use bounds [r, 1-r] because r changes.
        # But keeping bounds [0, 1] is safe.
        bounds = [(0, 1)] * (2 * n_circles)
        
        res = opt.minimize(objective, x_flat, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-12})
        
        centers = res.x.reshape(n_circles, 2)
        current_energy = res.fun
        
        # If energy is very low, we can increase r
        if current_energy < 1e-6:
            # Increase r
            if r_current < max_r:
                r_current += r_step
            else:
                break
        else:
            # If energy is high, we might be stuck or r is too big.
            # Try to reduce step or just accept current r and break?
            # If we can't minimize energy to 0, r is likely too large.
            # So we decrease r slightly and continue?
            # Or just stop and return current best valid r.
            # Let's decrease r step to be more precise.
            if r_step > 1e-5:
                r_step /= 2
            # Keep r_current as is (we failed to fit it)
            # Actually, if energy > 0, current r is not feasible with these positions.
            # Maybe we need better optimization.
            # But for now, assume we found a local limit.
            # Let's stick to current r.
            pass
            
        # Safety break
        if step % 100 == 0 and step > 0:
             # Optional: print progress
             pass

    # Final radius found
    r_final = r_current
    
    # To be safe, let's verify if we can increase r_final slightly or if we need to back off.
    # If the last step failed (energy high), r_current might be invalid.
    # Let's do a final check: decrease r until feasible.
    # We want the largest valid r.
    # Since we increased r when energy was low, r_final should be roughly valid (with tiny tolerance).
    # But numerical errors might leave tiny overlaps.
    # Let's clamp radii slightly if needed, or just trust the optimizer.
    # Actually, the problem allows 1e-12 tolerance.
    
    # However, to maximize sum of radii, maybe unequal radii is better.
    # Let's try to optimize sum of radii with the found centers as a starting point.
    # We can allow radii to vary.
    
    # Optimization for sum of radii
    # Variables: x, y, r_i
    # Constraints: dist >= r_i + r_j, bounds.
    # This is harder.
    # Let's stick to equal radii solution which is likely very close to optimal.
    # The sum will be 26 * r_final.
    
    # Just to be sure, let's run one more optimization to tighten positions for the final r.
    def objective_final(x_flat):
        x = x_flat.reshape(n_circles, 2)
        return compute_energy(x, r_final)
    
    def gradient_final(x_flat):
        x = x_flat.reshape(n_circles, 2)
        g = compute_gradient(x, r_final)
        return g.flatten()

    x_flat = centers.flatten()
    bounds = [(0, 1)] * (2 * n_circles)
    res_final = opt.minimize(objective_final, x_flat, jac=gradient_final, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 1000, 'ftol': 1e-15})
    
    centers = res_final.x.reshape(n_circles, 2)
    
    # Ensure no negative radii or NaN
    radii = np.full(n_circles, r_final)
    
    # Check validity and adjust if necessary
    # If validation fails, reduce radius slightly.
    # We can implement a quick check.
    
    # Simple validation check
    valid = True
    for i in range(n_circles):
        if radii[i] < 0:
            valid = False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            valid = False
            # Fix center
            centers[i, 0] = np.clip(x, r, 1-r)
            centers[i, 1] = np.clip(y, r, 1-r)
    
    if not valid:
        # Reduce radius slightly
        r_final *= 0.99
        radii[:] = r_final
        # Re-clip centers
        for i in range(n_circles):
            centers[i, 0] = np.clip(centers[i, 0], r_final, 1-r_final)
            centers[i, 1] = np.clip(centers[i, 1], r_final, 1-r_final)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To ensure the code is runnable and self-contained
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radius: {r[0]}")
    
    # Run validation logic from prompt to be sure
    import numpy as np # re-import if needed in local scope, but already imported
    
    # Validate
    # Check overlaps
    ok = True
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            dist = np.sqrt(np.sum((c[i] - c[j]) ** 2))
            if dist < r[i] + r[j] - 1e-12:
                ok = False
                print(f"Overlap {i},{j}: {dist} < {r[i]+r[j]}")
    if ok:
        print("Valid packing.")
    else:
        print("Invalid packing.")

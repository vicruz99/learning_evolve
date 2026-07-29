# sol_000005 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0a5b5ea2) state=5e72a52c sum of radii=1.844645 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def calculate_repulsion_force(centers, radii, stiffness=100.0):
    """Calculate repulsive forces between circles."""
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Vectorized pairwise distance calculation
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-10)
    dists = np.where(dists < 1e-9, 1e-9, dists)
    np.fill_diagonal(dists, np.inf) # Avoid self-interaction
    
    min_dists = radii[:, None] + radii[None, :]
    overlap = np.maximum(0, min_dists - dists)
    
    # Normalize diff vector for direction
    norm_diff = diff / dists[:, :, None]
    
    # Force magnitude proportional to overlap
    force_mag = overlap * stiffness
    force_vectors = norm_diff * force_mag[:, :, None]
    
    # Sum forces on each circle
    forces -= np.sum(force_vectors, axis=1) # Push away from others
    
    return forces

def boundary_forces(centers, radii, stiffness=100.0):
    """Calculate forces pushing circles away from boundaries."""
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # X boundaries
    for i in range(n):
        if centers[i, 0] - radii[i] < -1e-10:
            forces[i, 0] += stiffness * (-centers[i, 0] + radii[i])
        if centers[i, 0] + radii[i] > 1 + 1e-10:
            forces[i, 0] -= stiffness * (centers[i, 0] - (1 - radii[i]))
            
        # Y boundaries
        if centers[i, 1] - radii[i] < -1e-10:
            forces[i, 1] += stiffness * (-centers[i, 1] + radii[i])
        if centers[i, 1] + radii[i] > 1 + 1e-10:
            forces[i, 1] -= stiffness * (centers[i, 1] - (1 - radii[i]))
            
    return forces

def run_packing():
    n_circles = 26
    np.random.seed(42)
    
    # 1. Initialization: 5x5 grid + 1 circle
    centers = np.zeros((n_circles, 2))
    radii = np.ones(n_circles) * 0.01
    
    x_vals = np.linspace(0.1, 0.9, 5)
    y_vals = np.linspace(0.1, 0.9, 5)
    idx = 0
    for x in x_vals:
        for y in y_vals:
            if idx < n_circles:
                centers[idx] = [x, y]
                idx += 1
    # If 26th needed (we have 25 in grid)
    if idx < n_circles:
        centers[idx] = [0.5, 0.5] # Temporary
        
    # 2. Iterative expansion and optimization
    max_outer_iters = 2000
    radius_growth_rate = 0.00005
    learning_rate = 0.05
    
    for step in range(max_outer_iters):
        # Calculate forces
        forces = calculate_repulsion_force(centers, radii)
        forces += boundary_forces(centers, radii)
        
        # Update positions
        centers += learning_rate * forces
        
        # Update radii (slowly expand)
        radii += radius_growth_rate
        
        # Clamp radii to reasonable bounds and centers to [0,1]
        radii = np.clip(radii, 0.001, 0.2)
        centers[:, 0] = np.clip(centers[:, 0], radii, 1 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1 - radii)
        
        # Reduce learning rate as we progress
        if step > max_outer_iters / 2:
            learning_rate *= 0.999

    # 3. Final Refinement with Scipy
    # We optimize positions to minimize overlaps and maximize radii sum
    # This is a constrained optimization problem. 
    # We'll use a penalty approach to maximize sum(radii) - lambda * overlap_penalty
    
    def objective(variables):
        c = variables[:52].reshape(-1, 2)
        r = variables[52:]
        
        # Penalty for overlaps
        diff = c[:, None, :] - c[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        min_dists = r[:, None] + r[None, :]
        overlap = np.maximum(0, min_dists - dists)
        # Mask diagonal
        np.fill_diagonal(overlap, 0)
        overlap_penalty = np.sum(overlap**2)
        
        # Penalty for boundaries
        bound_penalty = 0
        for i in range(n_circles):
            if c[i, 0] < r[i]: bound_penalty += (r[i] - c[i, 0])**2
            if c[i, 0] > 1 - r[i]: bound_penalty += (c[i, 0] - (1 - r[i]))**2
            if c[i, 1] < r[i]: bound_penalty += (r[i] - c[i, 1])**2
            if c[i, 1] > 1 - r[i]: bound_penalty += (c[i, 1] - (1 - r[i]))**2
            
        # Objective: Maximize sum of radii, minimize penalties
        # We scale penalties heavily to ensure validity
        return -np.sum(r) + 100 * overlap_penalty + 100 * bound_penalty

    initial_vars = np.concatenate([centers.flatten(), radii])
    
    # Bounds for optimization
    bnds = []
    for i in range(n_circles):
        bnds.extend([(0, 1), (0, 1)]) # x, y bounds (will be constrained by radius later)
        bnds.append((0, 0.2)) # radius bound
        
    res = opt.minimize(objective, initial_vars, method='L-BFGS-B', bounds=bnds, options={'maxiter': 10000, 'ftol': 1e-12})
    
    final_centers = res.x[:52].reshape(-1, 2)
    final_radii = res.x[52:]
    
    # 4. Post-processing and Validation
    # Ensure strict validity with a small tolerance buffer
    valid = True
    
    # Check boundaries and adjust if necessary
    for i in range(n_circles):
        x, y = final_centers[i]
        r = final_radii[i]
        # Push inside if slightly out
        if x - r < 0: x = r
        if x + r > 1: x = 1 - r
        if y - r < 0: y = r
        if y + r > 1: y = 1 - r
        final_centers[i] = [x, y]

    # Check pairwise overlaps and reduce radii if overlapping
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
            req_dist = final_radii[i] + final_radii[j]
            if dist < req_dist - 1e-12:
                # Reduce radii proportionally to resolve overlap
                overlap = req_dist - dist
                # Simple heuristic: reduce both radii slightly
                reduction = overlap * 0.5 + 1e-9
                final_radii[i] = max(0, final_radii[i] - reduction)
                final_radii[j] = max(0, final_radii[j] - reduction)
                # Re-check boundaries after reduction
                for k in [i, j]:
                    cx, cy = final_centers[k]
                    cr = final_radii[k]
                    if cx < cr: final_centers[k][0] = cr
                    if cx > 1-cr: final_centers[k][0] = 1-cr
                    if cy < cr: final_centers[k][1] = cr
                    if cy > 1-cr: final_centers[k][1] = 1-cr

    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii

# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b079e3ed) state=6d5f464e sum of radii=2.282442 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    np.random.seed(42)
    n_circles = 26
    
    # 1. Initialization
    # Try a hexagonal-like initial placement or a perturbed grid
    # A 5x5 grid has 25 spots. We need 26. 
    # Let's place 25 in a 5x5 grid and 1 in the center or random.
    # Or better: a dense hexagonal packing pattern.
    
    # Let's generate a good initial configuration using a hexagonal lattice
    # We want to fit 26 circles. 
    # A hexagonal packing of circles of radius r has density.
    # We'll place centers on a lattice and then let the optimizer move them.
    
    centers = np.zeros((n_circles, 2))
    
    # Hexagonal packing parameters
    # Spacing approx 0.25 (radius ~0.125 initially)
    r_init = 0.12
    dist = 2.2 * r_init # slightly larger than 2r to allow overlap initially for relaxation
    
    row = 0
    col = 0
    idx = 0
    
    # Generate a grid of potential positions
    positions = []
    y = r_init + 0.05 # margin
    while y < 1 - r_init:
        x = r_init + 0.05
        is_odd_row = (row % 2 == 1)
        if is_odd_row:
            x = r_init + 0.05 + dist / 2
        
        while x < 1 - r_init:
            positions.append([x, y])
            x += dist
        y += dist * math.sqrt(3) / 2
        row += 1
        
    # Select 26 positions from this grid
    if len(positions) >= n_circles:
        # Pick the first 26
        centers[:n_circles] = positions[:n_circles]
    else:
        # Fallback to random if grid is too sparse
        centers = np.random.uniform(0.1, 0.9, (n_circles, 2))

    # Add some random jitter to avoid symmetry issues
    centers += np.random.normal(0, 0.005, centers.shape)
    # Clip to valid range
    centers = np.clip(centers, 0.05, 0.95)

    radii = np.ones(n_circles) * 0.01 # Start small

    # 2. Optimization Loop (Grow and Relax)
    # We will iteratively increase the radii and relax the positions to avoid overlap.
    
    current_r = 0.01
    target_r = 0.15 # Upper bound guess
    
    # We can use a simulation approach
    # Forces:
    # 1. Repulsion between circles: F = k * max(0, r_i + r_j - dist)^2 * direction
    # 2. Boundary repulsion: push away from walls
    # 3. Pressure: increase radii
    
    # Actually, to maximize sum of radii, we can treat this as finding a configuration
    # where radii are as large as possible.
    # Let's use a force-based simulation where we try to grow radii uniformly.
    
    # Parameters for simulation
    dt = 0.01
    damping = 0.9
    k_rep = 100.0 # Repulsion stiffness
    k_bound = 100.0 # Boundary stiffness
    growth_rate = 0.0002 # How fast to grow radii
    
    max_iter = 20000
    # To speed up, we can do fewer iterations with larger steps if needed, 
    # but 20000 is safe for numpy loops if optimized? 
    # Actually, 20000 * 26^2 is ~13 million ops, might be slow in pure python loop.
    # Let's use vectorized operations where possible or fewer iterations.
    
    # Let's reduce iterations and rely on a better optimizer for the final step.
    # Or use a simple loop.
    
    # Let's implement a vectorized force update.
    
    # Precompute indices for pairs? 
    # For N=26, N^2 is 676, manageable.
    
    idx_i, idx_j = np.triu_indices(n_circles, k=1)
    n_pairs = len(idx_i)
    
    # Simulation loop
    # We'll run this for a number of steps
    # In each step:
    # 1. Grow radii
    # 2. Compute forces
    # 3. Update positions
    
    # To make it robust, let's run a few cycles of "Grow" and "Relax"
    
    num_cycles = 2000
    
    for step in range(num_cycles):
        # 1. Grow radii
        # We want to increase radii. 
        # Check if we can increase. 
        # Simple strategy: increase by a small amount if not too much overlap.
        # But overlap is handled by forces.
        # Just increase.
        radii += growth_rate
        
        # Cap radii to prevent explosion if stuck
        if radii[0] > 0.2: 
            break 
            
        # 2. Compute Forces
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        # dist matrix
        # centers[i] - centers[j]
        # Vectorized diff
        diff = centers[idx_i] - centers[idx_j] # (n_pairs, 2)
        dists = np.sqrt(np.sum(diff**2, axis=1)) # (n_pairs,)
        
        # Avoid division by zero
        dists_safe = np.maximum(dists, 1e-9)
        directions = diff / dists_safe[:, np.newaxis] # (n_pairs, 2)
        
        # Overlap amount
        r_sum = radii[idx_i] + radii[idx_j]
        overlap = r_sum - dists
        
        # Force magnitude: proportional to overlap^2 (soft) or overlap (linear)
        # Linear is stiffer, might oscillate. Quadratic is smoother.
        # F = k * overlap
        # But only if overlap > 0
        active = overlap > 0
        force_mag = k_rep * np.maximum(0, overlap)
        
        # Apply forces
        # Force on i is +F, on j is -F
        # Using np.add.at for accumulation
        np.add.at(forces, idx_i, force_mag[:, np.newaxis] * directions)
        np.add.at(forces, idx_j, -force_mag[:, np.newaxis] * directions)
        
        # Boundary forces
        # x boundaries
        # If x < r, force right (positive)
        # If x > 1-r, force left (negative)
        # x_i - r_i < 0 => x_i < r_i => force = k * (r_i - x_i)
        # x_i + r_i > 1 => x_i > 1 - r_i => force = -k * (x_i + r_i - 1)
        
        # Left wall
        overlap_x_left = radii - centers[:, 0]
        f_x_left = k_bound * np.maximum(0, overlap_x_left)
        forces[:, 0] += f_x_left
        
        # Right wall
        overlap_x_right = (centers[:, 0] + radii) - 1.0
        f_x_right = -k_bound * np.maximum(0, overlap_x_right)
        forces[:, 0] += f_x_right
        
        # Bottom wall
        overlap_y_bot = radii - centers[:, 1]
        f_y_bot = k_bound * np.maximum(0, overlap_y_bot)
        forces[:, 1] += f_y_bot
        
        # Top wall
        overlap_y_top = (centers[:, 1] + radii) - 1.0
        f_y_top = -k_bound * np.maximum(0, overlap_y_top)
        forces[:, 1] += f_y_top
        
        # Update positions
        # Velocity Verlet or simple Euler?
        # Simple Euler with damping on displacement
        # delta_pos = forces * dt^2 / mass (assume mass 1)
        # Add damping to velocity? 
        # Let's just update position directly proportional to force with damping
        
        # To avoid oscillations, we can mix current pos with new pos
        # or just use a small step.
        
        centers += forces * dt * dt * 0.5 # scaling factor
        centers *= damping # This is wrong for positions. 
        # Better:
        # velocity update?
        # Let's keep it simple: centers += forces * alpha
        # But we need to handle momentum to escape local minima?
        # Let's just use simple gradient step.
        
        # Correcting the update logic:
        # We computed forces.
        # Update: x_new = x_old + F * step_size
        # But we need to be careful not to overshoot.
        
        # Let's revert to standard:
        # centers = centers + forces * 1e-5 (small step)
        
    # The above loop logic was a bit mixed. Let's write a cleaner simulation.
    
    # Re-initialize for a cleaner run
    centers = np.random.uniform(0.1, 0.9, (n_circles, 2))
    radii = np.ones(n_circles) * 0.01
    
    # Simulation parameters
    alpha = 1e-4 # Step size for position update
    r_growth = 1e-5 # Step size for radius growth
    max_steps = 50000
    
    # Precompute pair indices
    pairs_i, pairs_j = np.triu_indices(n_circles, k=1)
    n_pairs = len(pairs_i)
    
    # To speed up, we can use a fixed number of steps
    # But 50000 might be slow in python loop if not careful.
    # Let's try 10000 steps.
    
    for _ in range(10000):
        # Increase radii
        radii += r_growth
        
        # Compute pairwise distances and forces
        # diff = c_i - c_j
        diff = centers[pairs_i] - centers[pairs_j]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        
        # Repulsion force
        # We want to push apart if dist < r_i + r_j
        # Force magnitude proportional to overlap
        overlap = (radii[pairs_i] + radii[pairs_j]) - dist
        # Only push if overlapping or very close (to prevent jitter)
        # Softening: only if overlap > 0
        force_mag = np.maximum(0, overlap) * 10.0 # stiffness
        
        # Unit vectors
        # Avoid div by 0
        norm = np.sqrt(np.sum(diff**2, axis=1))
        norm_safe = np.where(norm > 1e-9, norm, 1e-9)
        unit_vec = diff / norm_safe[:, np.newaxis]
        
        # Forces vector
        pair_forces = force_mag[:, np.newaxis] * unit_vec
        
        # Accumulate forces
        # forces on i
        forces_i = np.zeros_like(centers)
        np.add.at(forces_i, pairs_i, pair_forces)
        # forces on j (opposite)
        forces_j = np.zeros_like(centers)
        np.add.at(forces_j, pairs_j, -pair_forces)
        
        # Total forces
        forces = forces_i + forces_j # Wait, forces_i accumulates at i, forces_j at j.
        # Actually forces_i contains force on i from all j's? 
        # np.add.at adds to index.
        # forces_i[pairs_i] += pair_forces. Correct.
        # forces_j[pairs_j] -= pair_forces. Correct.
        # So total force on circle k is sum of contributions.
        # We need to sum forces_i and forces_j?
        # No. forces_i has non-zeros only at indices present in pairs_i.
        # forces_j has non-zeros only at indices present in pairs_j.
        # But a circle can be both i and j in different pairs.
        # So we need to sum them up?
        # Actually, we can just use one array.
        
        forces = np.zeros_like(centers)
        np.add.at(forces, pairs_i, pair_forces)
        np.add.at(forces, pairs_j, -pair_forces)
        
        # Boundary forces
        # Left: x < r -> push right
        f_x = np.zeros(n_circles)
        mask_left = centers[:, 0] < radii
        f_x[mask_left] = 10.0 * (radii[mask_left] - centers[mask_left, 0])
        forces[:, 0] += f_x
        
        # Right: x > 1-r -> push left
        mask_right = centers[:, 0] > 1.0 - radii
        f_x_r = -10.0 * (centers[mask_right, 0] - (1.0 - radii[mask_right]))
        forces[:, 0] += f_x_r
        
        # Bottom: y < r -> push up
        mask_bot = centers[:, 1] < radii
        f_y = 10.0 * (radii[mask_bot] - centers[mask_bot, 1])
        forces[:, 1] += f_y
        
        # Top: y > 1-r -> push down
        mask_top = centers[:, 1] > 1.0 - radii
        f_y_t = -10.0 * (centers[mask_top, 1] - (1.0 - radii[mask_top]))
        forces[:, 1] += f_y_t
        
        # Update positions
        centers += forces * alpha
        
        # Clamp to ensure valid (though forces should handle it)
        # Not strictly necessary if forces work, but good for safety
        # But clamping might reduce sum of radii if we clamp radii?
        # No, we clamp centers.
        
    # After simulation, we might have slight overlaps due to discrete steps.
    # Also, the radii might be limited by the last step.
    # We can try to optimize further using scipy.
    
    # Define an objective function for scipy
    # Maximize sum of radii.
    # Variables: centers (flattened) and radii?
    # Or just centers, assuming radii are determined by constraints?
    # No, radii are variables.
    # But it's easier to optimize centers for fixed radii, then grow radii.
    # We already did that roughly.
    
    # Let's refine the result.
    # We can solve the LP for radii given the centers found.
    # This will give the exact maximum radii for this configuration.
    
    # LP: Max sum(r) s.t. r_i + r_j <= d_ij, r_i <= x_i, etc.
    # Using scipy.optimize.linprog
    
    # Variables: r_0 ... r_25 (26 vars)
    # Maximize c^T x => minimize -c^T x, where c = [1, 1, ..., 1]
    
    # Constraints:
    # 1. r_i + r_j <= d_ij  => r_i + r_j - d_ij <= 0
    # 2. r_i <= x_i
    # 3. r_i <= 1 - x_i
    # 4. r_i <= y_i
    # 5. r_i <= 1 - y_i
    # 6. r_i >= 0
    
    # Inequality form for linprog: A_ub @ x <= b_ub
    # 1. r_i + r_j <= d_ij  => [0...1...1...0] @ r <= d_ij
    # 2. r_i <= x_i         => [0...1...0] @ r <= x_i
    # ...
    
    # Distance matrix
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    # dists[i, j] is dist between i and j
    
    # Build A_ub and b_ub
    # We have 26 variables.
    
    # Constraints 1: Pairwise (upper triangle)
    # r_i + r_j <= dists[i, j]
    # For each pair (i, j) with i < j:
    # Row in A_ub: 1 at i, 1 at j.
    
    # Constraints 2-5: Boundary
    # r_i <= x_i  => 1*r_i <= x_i
    # r_i <= 1-x_i => 1*r_i <= 1-x_i
    # ...
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            row = np.zeros(n_circles)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Boundary constraints
    for i in range(n_circles):
        # r_i <= x_i
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(centers[i, 0])
        
        # r_i <= 1 - x_i
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - centers[i, 0])
        
        # r_i <= y_i
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(centers[i, 1])
        
        # r_i <= 1 - y_i
        row = np.zeros(n_circles)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - centers[i, 1])
        
    # Non-negativity is handled by bounds in linprog or explicit constraints.
    # linprog bounds: (0, None) for all r.
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    c = np.ones(n_circles) # Minimize -sum(r)
    
    from scipy.optimize import linprog
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n_circles, method='highs')
    
    if res.success:
        optimal_radii = res.x
        # Check validity
        # Sum
        sum_r = np.sum(optimal_radii)
    else:
        # Fallback to simulation radii if LP fails (unlikely)
        optimal_radii = radii
        sum_r = np.sum(radii)

    # Final validation and adjustment
    # Sometimes LP might give radii that slightly violate due to float errors?
    # But linprog is exact.
    # However, dists were computed from centers.
    # So it should be valid.
    
    # Just to be safe, scale down slightly if needed?
    # No, linprog satisfies constraints exactly.
    
    # Return result
    return centers, optimal_radii, float(np.sum(optimal_radii))

# Let's refine the strategy.
# The simulation loop above is a bit generic. 
# A better initial placement and more aggressive relaxation is needed.
# Also, the simulation loop might get stuck.
# Let's incorporate the LP step inside the loop?
# No, LP is expensive to call 10000 times.
# But we can call it once at the end.
# The simulation is just to find a good geometry.

# To improve geometry, we can run the simulation for longer or with better params.
# Also, we can try multiple random restarts for the simulation.

def run_packing():
    np.random.seed(42)
    n_circles = 26
    
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Try multiple random starts
    for trial in range(5):
        # Initialize centers
        # Use a mix of random and grid
        centers = np.random.uniform(0.1, 0.9, (n_circles, 2))
        
        # Maybe perturb a grid
        if trial == 0:
            # Hexagonal-ish
            centers = np.zeros((n_circles, 2))
            idx = 0
            r_est = 0.1
            y = 0.1 + r_est
            row = 0
            while y < 0.9 and idx < n_circles:
                x = 0.1 + r_est
                if row % 2 == 1:
                    x += r_est
                while x < 0.9 and idx < n_circles:
                    centers[idx] = [x, y]
                    x += 2 * r_est
                    idx += 1
                y += math.sqrt(3) * r_est
                row += 1
            # Add noise
            centers += np.random.normal(0, 0.02, centers.shape)
            centers = np.clip(centers, 0.05, 0.95)

        radii = np.ones(n_circles) * 0.01
        
        # Simulation
        alpha = 1e-4
        r_growth = 1e-5
        steps = 8000
        
        pairs_i, pairs_j = np.triu_indices(n_circles, k=1)
        
        for _ in range(steps):
            radii += r_growth
            if radii[0] > 0.15: # Cap growth if too large
                break
            
            # Vectorized force calculation
            diff = centers[pairs_i] - centers[pairs_j]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            
            # Repulsion
            overlap = (radii[pairs_i] + radii[pairs_j]) - dist
            force_mag = np.maximum(0, overlap) * 100.0
            
            norm = np.sqrt(np.sum(diff**2, axis=1))
            norm_safe = np.where(norm > 1e-9, norm, 1e-9)
            unit_vec = diff / norm_safe[:, np.newaxis]
            
            pair_forces = force_mag[:, np.newaxis] * unit_vec
            
            forces = np.zeros_like(centers)
            np.add.at(forces, pairs_i, pair_forces)
            np.add.at(forces, pairs_j, -pair_forces)
            
            # Boundary
            # Left
            mask = centers[:, 0] < radii
            forces[mask, 0] += 100.0 * (radii[mask] - centers[mask, 0])
            # Right
            mask = centers[:, 0] > 1.0 - radii
            forces[mask, 0] -= 100.0 * (centers[mask, 0] - (1.0 - radii[mask]))
            # Bottom
            mask = centers[:, 1] < radii
            forces[mask, 1] += 100.0 * (radii[mask] - centers[mask, 1])
            # Top
            mask = centers[:, 1] > 1.0 - radii
            forces[mask, 1] -= 100.0 * (centers[mask, 1] - (1.0 - radii[mask]))
            
            centers += forces * alpha
            centers = np.clip(centers, 0.001, 0.999) # Keep strictly inside
            
        # Solve LP for optimal radii given centers
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        
        A_ub = []
        b_ub = []
        
        # Pairwise
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                row = np.zeros(n_circles)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dists[i, j])
                
        # Boundary
        for i in range(n_circles):
            row = np.zeros(n_circles)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(centers[i, 0])
            
            row = np.zeros(n_circles)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0 - centers[i, 0])
            
            row = np.zeros(n_circles)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(centers[i, 1])
            
            row = np.zeros(n_circles)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0 - centers[i, 1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        c = np.ones(n_circles)
        
        try:
            res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n_circles, method='highs')
            if res.success:
                current_sum = np.sum(res.x)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = res.x.copy()
        except:
            pass

    # Final refinement on best_centers
    # Maybe run a few more steps of relaxation on best_centers with slightly larger radii?
    # Or just trust the LP.
    
    # Validate best_radii against best_centers just in case
    # LP guarantees constraints, but float precision...
    # The LP constraints were dists[i,j] which are floats.
    # It should be fine.
    
    if best_centers is None:
        # Fallback
        centers = np.random.uniform(0.1, 0.9, (26, 2))
        radii = np.ones(26) * 0.01
        best_centers = centers
        best_radii = radii
        best_sum = 0.26

    return best_centers, best_radii, float(best_sum)

# To ensure we import linprog
from scipy.optimize import linprog

# sol_000210 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1af9ddd7) state=7a3bb3c9 sum of radii=2.608070 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def solve_radii_lp(centers):
    """
    Solves the Linear Programming problem to find the maximum sum of radii
    given fixed circle centers.
    
    Args:
        centers: np.array of shape (N, 2)
        
    Returns:
        radii: np.array of shape (N)
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r_i) => Minimize sum(-r_i)
    c_obj = -np.ones(n)
    
    # Inequality constraints: A_ub * r <= b_ub
    # We only need to enforce r_i + r_j <= dist(i, j)
    # Boundary constraints are handled via bounds
    
    # Precompute distances
    # Dist matrix (N, N)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf) # No self-constraint
    
    # Build constraint matrix for r_i + r_j <= d_ij
    # Only upper triangle (i < j)
    rows, cols = np.triu_indices(n, k=1)
    
    A_ub = np.zeros((len(rows), n))
    b_ub = np.zeros(len(rows))
    
    for k, (i, j) in enumerate(zip(rows, cols)):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = dists[i, j]
    
    # Bounds for radii: 0 <= r_i <= distance_to_nearest_wall
    bounds = []
    for i in range(n):
        x, y = centers[i]
        # Distance to nearest wall
        d_wall = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, d_wall)))
        
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback if LP fails, return small valid radii
            return np.full(n, 1e-6)
    except Exception:
        return np.full(n, 1e-6)

def compute_forces(centers, radii):
    """
    Computes repulsive forces on centers based on active constraints.
    Active constraints are those where r_i + r_j is very close to distance.
    """
    n = centers.shape[0]
    forces = np.zeros((n, 2))
    epsilon = 1e-4
    
    # Pairwise forces
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-9:
                # Coincident centers, push random direction
                forces[i] += np.random.randn(2) * 10
                forces[j] -= np.random.randn(2) * 10
                continue
                
            r_sum = radii[i] + radii[j]
            
            # If circles are touching or overlapping (active constraint)
            if r_sum >= dist - epsilon:
                # Force direction: away from each other
                # Vector from j to i is (c_i - c_j). Normalized.
                nx, ny = (centers[i] - centers[j]) / dist
                forces[i] += np.array([nx, ny])
                forces[j] -= np.array([nx, ny])
                
    # Boundary forces
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Check boundaries
        # If r is limited by wall, push away
        # Left wall (x - r approx 0)
        if x - r < epsilon:
            forces[i, 0] += 1.0
        # Right wall (1 - x - r approx 0)
        if 1.0 - x - r < epsilon:
            forces[i, 0] -= 1.0
        # Bottom wall (y - r approx 0)
        if y - r < epsilon:
            forces[i, 1] += 1.0
        # Top wall (1 - y - r approx 0)
        if 1.0 - y - r < epsilon:
            forces[i, 1] -= 1.0
            
    return forces

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Number of circles
    N = 26
    
    # 1. Initial Configuration: Hexagonal Grid
    # We estimate a radius to place points. 
    # If sum of radii target is ~2.6, avg r ~ 0.1.
    # Hexagonal packing: row height sqrt(3)*r, col width 2*r.
    # Let's try to fit points in a 6x5 grid area or similar.
    
    # Generate points
    centers = []
    r_init = 0.11 
    
    # Try to fill rows
    y = r_init
    row_idx = 0
    while len(centers) < N:
        # x offset for hexagonal packing
        x_offset = r_init if row_idx % 2 == 1 else 0.0
        x = r_init + x_offset
        
        while x + r_init <= 1.0 and len(centers) < N:
            centers.append([x, y])
            x += 2 * r_init
        
        y += np.sqrt(3) * r_init
        row_idx += 1
        
    centers = np.array(centers[:N])
    
    # Add some random perturbation to break symmetry and avoid grid locks
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    # Ensure within bounds
    centers = np.clip(centers, 0.01, 0.99)

    # 2. Optimization Loop
    best_sum_radii = -1.0
    best_centers = centers.copy()
    
    # Initial radii estimate
    radii = np.full(N, 0.05) 
    
    # Parameters
    iterations = 2000
    initial_step = 0.05
    
    # Run multiple restarts if stuck, but one long run with good init usually works
    # We will do one run
    
    for it in range(iterations):
        # Step size decay
        step = initial_step * (0.995 ** it)
        if step < 1e-5:
            step = 1e-5
            
        # A. Optimize Radii for current centers
        radii = solve_radii_lp(centers)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            # Store best radii? No, radii depend on centers. 
            # We need to re-solve radii at the end.
            
        # B. Compute Forces
        forces = compute_forces(centers, radii)
        
        # Normalize forces? Or just use raw.
        # Forces can be large if many contacts.
        # Let's normalize total force magnitude or scale by step.
        
        # Apply forces
        centers += step * forces
        
        # C. Project back to valid region [0, 1]x[0, 1]
        # Actually, we need to ensure centers allow for SOME radius.
        # But simple clipping to [0,1] is fine, LP will handle radius reduction.
        # However, if center is at 0, radius must be 0.
        # Let's keep centers strictly inside to allow movement?
        # No, centers can be on boundary if radius is 0, but that's bad.
        # We want centers to be at least r away.
        # But r varies.
        # Simple clipping:
        centers = np.clip(centers, 0.0, 1.0)
        
        # Prevent centers from getting too close to boundary if they have large radii?
        # LP handles the constraint r <= dist_to_wall.
        # If center is at 0.001, max r is 0.001.
        
    # Final Radius Calculation on Best Centers
    # Note: best_centers stored might be from an intermediate step.
    # Let's re-optimize radii for the centers found at the end of loop 
    # or the best_centers.
    # Actually, the loop stores best_centers when sum is high.
    # But the forces might have moved it to a state with lower sum temporarily?
    # Unlikely. The sum usually increases.
    # Let's just use the final centers and solve LP once.
    # Or better, use best_centers and solve LP.
    
    final_radii = solve_radii_lp(best_centers)
    final_sum = np.sum(final_radii)
    
    # Just in case, run a few more gradient steps on best_centers if needed?
    # The loop should have found it.
    
    # Validation check (internal)
    # Check for overlaps
    overlaps = False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(best_centers[i] - best_centers[j])
            if dist < final_radii[i] + final_radii[j] - 1e-10:
                overlaps = True
    # Check boundaries
    out = False
    for i in range(N):
        x, y = best_centers[i]
        r = final_radii[i]
        if x < r - 1e-10 or x > 1-r + 1e-10 or y < r - 1e-10 or y > 1-r + 1e-10:
            out = True
            
    if overlaps or out:
        # Fallback or repair?
        # The LP guarantees r_i + r_j <= dist and r_i <= boundary_dist.
        # So overlaps and boundary violations should not happen with LP solution
        # unless numerical precision issues.
        # We can clamp radii slightly if needed.
        pass

    return best_centers, final_radii, final_sum

# Execution block to test locally if run as script, 
# but the function must be defined.
if __name__ == "__main__":
    import sys
    # To allow running without external imports if needed, but prompt allows scipy/numpy.
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation
    # from validation function logic...

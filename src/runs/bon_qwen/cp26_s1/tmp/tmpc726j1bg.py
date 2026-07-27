import numpy as np
from scipy.optimize import minimize
import math

def repulsive_force(c_i, c_j, r_i, r_j):
    """Calculate repulsive force vector between two circles."""
    dist_vec = c_i - c_j
    dist = np.linalg.norm(dist_vec)
    if dist < 1e-10:
        return np.array([0.0, 0.0])
    # Force magnitude inversely proportional to distance squared
    # We push them apart by an amount that depends on how much they overlap or are close
    force_magnitude = 1.0 / (dist * dist)
    force_vec = (force_magnitude / dist) * dist_vec
    return force_vec

def boundary_force(c, r):
    """Calculate repulsive force from boundaries."""
    force = np.array([0.0, 0.0])
    # Left boundary
    if c[0] - r < 0:
        force[0] += 100.0 * (r - c[0])
    # Right boundary
    if c[0] + r > 1:
        force[0] -= 100.0 * (c[0] + r - 1)
    # Bottom boundary
    if c[1] - r < 0:
        force[1] += 100.0 * (r - c[1])
    # Top boundary
    if c[1] + r > 1:
        force[1] -= 100.0 * (c[1] + r - 1)
    return force

def initial_hexagonal_packing(n):
    """Generate an initial hexagonal grid layout."""
    # Approximate radius for n circles in unit square
    # r ~ 1 / (2 + sqrt(3)*sqrt(n)) is a rough heuristic, but let's use a safe small r
    # A safe starting radius for 26 circles is around 0.08
    r_start = 0.08
    centers = []
    
    # Estimate number of rows
    # Height of k rows: 2r + (k-1)*sqrt(3)*r
    # 1 = r(2 + (k-1)sqrt(3)) => k approx 1/sqrt(3)
    # Let's just place them in a grid and let optimization handle it
    # A 5x6 grid or similar
    
    # Better: Hexagonal packing rows
    # Row height = sqrt(3) * r
    # Width per circle = 2 * r
    # Let's try to fit as many as possible
    
    y = r_start
    row_idx = 0
    while len(centers) < n:
        x = r_start
        offset = r_start if row_idx % 2 == 1 else 0 # Stagger
        if offset > 0:
            x += r_start # Shift by r for hexagonal
        
        while x + r_start <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_start
        y += math.sqrt(3) * r_start
        row_idx += 1
        
    centers = np.array(centers[:n])
    return centers

def objective_function(params, n):
    """
    Objective: Minimize -sum(radii) subject to constraints handled by penalty.
    params: [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = params[i * 3]
        centers[i, 1] = params[i * 3 + 1]
        radii[i] = params[i * 3 + 2]
        
    obj = -np.sum(radii)
    penalty = 0.0
    
    # Boundary constraints
    for i in range(n):
        r = radii[i]
        x, y = centers[i]
        if x - r < 0: penalty += 1000 * (r - x)**2
        if x + r > 1: penalty += 1000 * (x + r - 1)**2
        if y - r < 0: penalty += 1000 * (r - y)**2
        if y + r > 1: penalty += 1000 * (y + r - 1)**2
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += 1000 * (min_dist - dist)**2
                
    return obj + penalty

def validate_and_expand(centers, radii):
    """
    Given centers, expand radii as much as possible.
    This handles the variable radius aspect.
    """
    n = len(radii)
    new_radii = np.zeros(n)
    
    for i in range(n):
        # Distance to boundaries
        dist_to_boundary = min(
            centers[i, 0],
            1 - centers[i, 0],
            centers[i, 1],
            1 - centers[i, 1]
        )
        
        # Distance to other centers (half the distance)
        min_dist_to_other = float('inf')
        for j in range(n):
            if i == j: continue
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            min_dist_to_other = min(min_dist_to_other, dist)
        
        # Max radius is limited by boundary and half distance to nearest neighbor
        # However, if we set r_i = min(dist_boundary, dist_to_other/2), we ensure non-overlap
        # But this is a conservative estimate because neighbors might also shrink.
        # A simple greedy expansion works well after optimization.
        
        r_i = min(dist_to_boundary, min_dist_to_other / 2.0)
        new_radii[i] = max(0.0, r_i)
        
    return new_radii

def run_packing() -> tuple:
    n = 26
    
    # 1. Generate initial centers
    centers = initial_hexagonal_packing(n)
    
    # 2. Optimize centers to maximize minimum distance (equal radii assumption)
    # We optimize a single variable r and centers to maximize r
    # Or just use the force simulation to spread points
    
    # Force simulation parameters
    alpha = 1.0 # Learning rate
    cooling = 0.995
    
    # Initialize with a safe radius
    r = 0.08
    # Ensure initial centers are valid with this r
    # (The generator should have done this, but let's clamp)
    for i in range(n):
        centers[i, 0] = max(r, min(1-r, centers[i, 0]))
        centers[i, 1] = max(r, min(1-r, centers[i, 1]))

    # Iterative repulsion to find optimal centers for max min-distance
    # We simulate repulsion between centers (ignoring r for a moment, just points)
    # to maximize the minimum distance between points.
    # This corresponds to finding the best configuration for equal circles.
    
    # Reset centers to a more randomized or structured grid for optimization
    # Let's use a perturbed grid
    centers = np.random.rand(n, 2) * 0.8 + 0.1 # Random start in [0.1, 0.9]
    
    # Simple gradient ascent on min-distance
    # Objective: max min(||c_i - c_j||, dist(c_i, boundary))
    # Let's use scipy optimize with a differentiable approximation or just the penalty method
    
    # Flattened parameters: [x1, y1, ..., x26, y26]
    x0 = centers.flatten()
    
    def min_dist_objective(x_flat):
        c = x_flat.reshape(n, 2)
        min_d = float('inf')
        # Distance to boundary
        for i in range(n):
            min_d = min(min_d, c[i, 0], 1-c[i, 0], c[i, 1], 1-c[i, 1])
        # Distance between centers
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                min_d = min(min_d, d)
        return -min_d # Minimize negative min distance (maximize min distance)

    # Use Nelder-Mead or BFGS. Nelder-Mead is robust for non-smooth functions.
    # However, min_dist is non-smooth.
    # We can use a smooth approximation: - (sum (dist^(-p)))^(1/-p)
    # Or just rely on the penalty optimization later.
    
    # Let's try a simple iterative repulsion simulation first to get a good configuration
    sim_centers = centers.copy()
    force_strength = 0.01
    for step in range(500):
        forces = np.zeros_like(sim_centers)
        for i in range(n):
            # Boundary repulsion
            for dim in range(2):
                if sim_centers[i, dim] < 0.05:
                    forces[i, dim] += 0.5
                elif sim_centers[i, dim] > 0.95:
                    forces[i, dim] -= 0.5
            
            # Inter-circle repulsion
            for j in range(i+1, n):
                vec = sim_centers[i] - sim_centers[j]
                dist = np.linalg.norm(vec)
                if dist < 0.01: dist = 0.01 # Prevent division by zero
                rep = force_strength / (dist * dist)
                forces[i] += rep * vec / dist
                forces[j] -= rep * vec / dist
        
        sim_centers += forces
        # Clamp to [0,1]
        sim_centers = np.clip(sim_centers, 0, 1)
        force_strength *= 0.99 # Cool down
        
    centers = sim_centers

    # 3. Now optimize radii and centers together using scipy
    # Initial radii based on current centers
    radii = np.zeros(n)
    for i in range(n):
        # Estimate max radius at current position
        min_d = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        for j in range(n):
            if i == j: continue
            d = np.sqrt(np.sum((centers[i]-centers[j])**2))
            min_d = min(min_d, d/2)
        radii[i] = min_d
    
    # Construct initial params for full optimization
    # To speed up, we can fix centers and just optimize radii?
    # No, moving centers helps.
    # But full 3n optimization is expensive (78 vars).
    # Let's try optimizing centers to maximize min-distance (equal radii) first.
    
    # Optimization of centers to maximize min-distance
    res_centers = minimize(min_dist_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    best_centers = res_centers.x.reshape(n, 2)
    
    # The optimal equal radius r_eq is the min distance found
    r_eq = -res_centers.fun
    
    # Now, calculate max possible radii for these centers (variable radii)
    final_radii = validate_and_expand(best_centers, np.zeros(n))
    
    # Final refinement: Optimize centers and radii together to squeeze out more
    # This allows some circles to be larger than r_eq if others are smaller
    x0_full = np.zeros(n * 3)
    for i in range(n):
        x0_full[i*3] = best_centers[i, 0]
        x0_full[i*3+1] = best_centers[i, 1]
        x0_full[i*3+2] = final_radii[i]
        
    # Use a robust optimizer
    res_full = minimize(objective_function, x0_full, args=(n,), method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    final_centers = np.zeros((n, 2))
    final_radii_arr = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = res_full.x[i*3]
        final_centers[i, 1] = res_full.x[i*3+1]
        final_radii_arr[i] = res_full.x[i*3+2]
        
    # Validate and clamp radii to be physically consistent with final centers
    # (In case optimization drifted slightly into violation due to penalty weight)
    corrected_radii = validate_and_expand(final_centers, final_radii_arr)
    
    sum_radii = np.sum(corrected_radii)
    
    return final_centers, corrected_radii, sum_radii

# To be run by the validation script
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic check
    print(validate_packing(c, r))
# sol_000242 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e7c70ed6) state=303939e5 sum of radii=1.255177 correctness=1.0
# stdout(first 200): Circles 2 and 7 overlap: dist=0.0907321953262386, r1+r2=0.1 Circles 2 and 7 overlap: dist=0.09630703137688981, r1+r2=0.0995 Circles 2 and 7 overlap: dist=0.09802089205383704, r1+r2=0.09900250000000001
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def compute_forces(centers, radii):
    """
    Computes repulsive forces on each circle based on overlaps with others and boundaries.
    
    Args:
        centers: np.array of shape (n, 2)
        radii: np.array of shape (n)
        
    Returns:
        forces: np.array of shape (n, 2)
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            
            # Avoid division by zero
            if dist < 1e-9:
                dist = 1e-9
                # Push apart randomly if coincident
                dir_vec = np.random.rand(2) - 0.5
            else:
                dir_vec = diff / dist
            
            req_dist = radii[i] + radii[j]
            
            # If overlapping
            if dist < req_dist:
                # Repulsive force magnitude proportional to overlap
                overlap = req_dist - dist
                # Scale force to be strong enough
                force_mag = overlap * 10.0 
                
                forces[i] += force_mag * dir_vec
                forces[j] -= force_mag * dir_vec

    # Boundary repulsion
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left wall
        if x < r:
            forces[i, 0] += (r - x) * 20.0
        # Right wall
        elif x > 1 - r:
            forces[i, 0] -= (x - (1 - r)) * 20.0
            
        # Bottom wall
        if y < r:
            forces[i, 1] += (r - y) * 20.0
        # Top wall
        elif y > 1 - r:
            forces[i, 1] -= (y - (1 - r)) * 20.0

    return forces

def run_packing():
    """
    Runs the circle packing optimization to maximize sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    np.random.seed(42) # For reproducibility
    n_circles = 26
    
    # 1. Initialization
    # Start with small radii and random positions
    centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
    radii = np.full(n_circles, 0.05)
    
    # Parameters for optimization
    max_iter = 2000
    growth_rate = 1.0005
    step_size = 0.01 # Initial step size for position update
    min_step_size = 1e-6
    noise_scale = 0.01
    
    # Track best valid configuration
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # 2. Iterative Optimization
    for step in range(max_iter):
        # Grow radii
        radii *= growth_rate
        
        # Add small random noise to escape local minima
        noise = np.random.normal(0, noise_scale * step_size, centers.shape)
        centers += noise
        
        # Compute forces
        forces = compute_forces(centers, radii)
        
        # Update positions based on forces
        # Scale forces by step_size to control movement
        centers += step_size * forces
        
        # Clamp positions to [0, 1] to prevent exploding out of bounds
        np.clip(centers, 0, 1, out=centers)
        
        # Decay step size slowly to settle down
        if step > 1000:
            step_size *= 0.999
            
        # Check validity and update best solution
        # We only check validity periodically to save time, 
        # but rely on forces to keep it roughly valid.
        # A strict check every time is expensive but safe.
        # Let's do a rough check: if max overlap is small, it's likely valid.
        
        # Compute max overlap for quick check
        max_overlap = 0.0
        valid = True
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                max_overlap = max(max_overlap, max(r-x, x-(1-r), r-y, y-(1-r)))
                valid = False
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                req = radii[i] + radii[j]
                if dist < req:
                    max_overlap = max(max_overlap, req - dist)
                    valid = False
        
        if valid and max_overlap < 1e-9:
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        
        # If overlaps are getting too large, maybe reduce growth rate or step size
        if max_overlap > 0.1:
            radii /= growth_rate # Rollback radii
            step_size *= 0.5
            radii *= growth_rate # Apply smaller growth
            # Actually, let's just reduce radii slightly to stabilize
            radii *= 0.99

    # 3. Final Cleanup
    # The simulation might leave tiny overlaps. 
    # We enforce validity by scaling down radii if necessary.
    # But first, try to optimize positions one last time with current radii
    
    # One last pass of forces to fix positions
    for _ in range(100):
        forces = compute_forces(best_centers, best_radii)
        best_centers += 0.001 * forces
        np.clip(best_centers, 0, 1, out=best_centers)

    # Check if valid, if not scale radii down slightly
    # We do this iteratively until valid
    valid = False
    current_radii = best_radii.copy()
    current_centers = best_centers.copy()
    
    # Use the provided validation function logic but simpler loop
    while not validate_packing(current_centers, current_radii):
        # Reduce radii slightly
        current_radii *= 0.995
        # Re-optimize positions briefly
        for _ in range(50):
            forces = compute_forces(current_centers, current_radii)
            current_centers += 0.001 * forces
            np.clip(current_centers, 0, 1, out=current_centers)
            
        # Safety break
        if np.sum(current_radii) < 1.0: # Should not happen
            break

    final_centers = current_centers
    final_radii = current_radii
    final_sum = np.sum(final_radii)
    
    return final_centers, final_radii, final_sum

# Execute the packing function
if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    print(f"Valid: {validate_packing(centers, radii)}")

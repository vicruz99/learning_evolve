# sol_000072 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 34f92e2c) state=2fb16bad sum of radii=2.081300 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import random
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square.
    Uses a force-directed local search approach with iterative expansion.
    """
    n_circles = 26
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # 1. Initialization: 5x5 grid plus center
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    base_r = 0.08 # Start slightly smaller than 0.1 to ensure valid start
    
    idx = 0
    # Create a 5x5 grid
    for r_idx in range(5):
        for c_idx in range(5):
            x = 0.1 + c_idx * 0.2 + (random.random() - 0.5) * 0.01
            y = 0.1 + r_idx * 0.2 + (random.random() - 0.5) * 0.01
            centers[idx] = [x, y]
            radii[idx] = base_r
            idx += 1
            
    # Add the 26th circle at the center with slightly smaller radius
    centers[idx] = [0.5, 0.5]
    radii[idx] = base_r * 0.9
    
    # 2. Optimization Loop
    # We will try to expand the radii while pushing circles apart
    num_iterations = 5000
    dt = 0.01
    friction = 0.95
    
    # Velocity array
    velocities = np.random.normal(0, 0.01, size=(n_circles, 2))
    
    # Target radius to aim for (gradually increase)
    # We'll let the optimizer find the max radius naturally by pushing boundaries
    # Instead of fixed radii, we will treat radius as a variable that grows 
    # if space permits, but here we'll just expand all equally.
    
    current_r = base_r
    
    for step in range(num_iterations):
        forces = np.zeros_like(centers)
        
        # 1. Repulsive forces between circles
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < 1e-9:
                    dist = 1e-9
                    dist_vec = np.random.normal(0, 0.001, 2)
                
                # Overlap amount
                if dist < min_dist:
                    # Force proportional to overlap, inverse square distance
                    overlap = min_dist - dist
                    force_mag = overlap * 10.0 # Stiffness
                    force_vec = (dist_vec / dist) * force_mag
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        # 2. Boundary forces (push circles away from walls)
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * 100.0
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 100.0
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * 100.0
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 100.0
            
            # Gentle outward pressure to expand radius
            # We simulate expansion by pushing circles toward walls slightly
            # If a circle is touching a wall, it can't expand. 
            # So we push them to corners/boundaries to make room?
            # Actually, simply increasing radius is better handled by a separate step.
            
        # Apply forces
        velocities += forces * dt
        velocities *= friction
        centers += velocities * dt
        
        # Clamp centers to valid range (soft clamp to allow solver to fix overlaps)
        # But for visualization/validity, we want them inside.
        # The forces above handle this.
        
        # 3. Iterative Expansion
        # Every 100 steps, try to increase radii slightly
        if step % 50 == 0:
            # Check if we can increase radius
            # Simple heuristic: if total overlap force is low, increase radius
            total_repulsion = 0
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < radii[i] + radii[j] + 1e-4:
                        total_repulsion += 1
            
            if total_repulsion == 0:
                # No significant overlaps, expand
                current_r += 0.00005
                # Update radii
                radii[:] = current_r
                # Add a tiny bit of random noise to positions to help escape local minima
                centers += np.random.normal(0, 0.001, size=centers.shape)

    # 4. Final Cleanup / Local Refinement
    # After the main loop, perform a few steps of pure radius maximization with fixed topology
    # Or just run a final pass to ensure validity
    
    # Ensure no circles are outside
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        # Pull back inside if necessary
        if x - r < 0: centers[i, 0] = r
        if x + r > 1: centers[i, 0] = 1 - r
        if y - r < 0: centers[i, 1] = r
        if y + r > 1: centers[i, 1] = 1 - r

    # Final validation check and correction
    # If overlaps persist, reduce radii slightly
    valid = False
    attempts = 0
    while not valid and attempts < 100:
        valid = True
        max_overlap = 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                required_dist = radii[i] + radii[j]
                if dist < required_dist:
                    valid = False
                    overlap = required_dist - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
        
        if not valid:
            # Reduce radii slightly to resolve
            radii[:] -= max_overlap * 0.5
            # Clamp radii to non-negative
            radii[:] = np.maximum(radii, 1e-6)
        
        # Re-check boundaries
        for i in range(n_circles):
            r = radii[i]
            x, y = centers[i]
            if x - r < 0: centers[i, 0] = r
            if x + r > 1: centers[i, 0] = 1 - r
            if y - r < 0: centers[i, 1] = r
            if y + r > 1: centers[i, 1] = 1 - r
            
        attempts += 1

    sum_radii = np.sum(radii)
    
    # Debug print for strategy verification
    # print(f"Final Sum of Radii: {sum_radii}")
    
    return centers, radii, sum_radii

# To ensure the function is callable and returns the correct types
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Generated packing with sum of radii: {s}")
    print(f"Number of circles: {len(r)}")
    print(f"Centers shape: {c.shape}")
    print(f"Radii shape: {r.shape}")

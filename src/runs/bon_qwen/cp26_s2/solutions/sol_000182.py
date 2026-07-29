# sol_000182 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0ae2e142) state=13a838d8 sum of radii=0.396238 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Uses a force-directed approach with pressure expansion.
    """
    n = 26
    np.random.seed(42)
    
    # 1. Initialization: 5x5 Grid + 1 Center
    centers = np.zeros((n, 2))
    # Place 25 circles in a grid
    idx = 0
    for r in range(5):
        for c in range(5):
            centers[idx, 0] = 0.1 + c * 0.2
            centers[idx, 1] = 0.1 + r * 0.2
            idx += 1
    # Place 26th circle in the center
    centers[25, 0] = 0.5
    centers[25, 1] = 0.5
    
    # Initial radii (small enough to not overlap initially)
    radii = np.ones(n) * 0.02

    # 2. Optimization Loop
    # Parameters
    dt = 0.01          # Time step
    repulsion_strength = 100.0
    boundary_strength = 100.0
    expansion_rate = 0.0001 # How much radii grow per step
    damping = 0.9      # Velocity damping
    iterations = 3000  # Total iterations
    
    velocities = np.zeros_like(centers)
    
    # Phase 1: High temperature to explore and separate overlaps
    # Phase 2: Cooling down to settle
    for i in range(iterations):
        # Temperature schedule: decay over time
        temp = 1.0 * (1.0 - i / iterations) + 0.01
        
        # Current expansion rate decreases slightly as we converge
        current_expansion = expansion_rate * (1.0 + 0.5 * temp)

        # Reset forces
        forces = np.zeros_like(centers)

        # Calculate pairwise repulsion
        for j in range(n):
            for k in range(j + 1, n):
                diff = centers[j] - centers[k]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    dist = 1e-8
                    diff = np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)])
                
                overlap = radii[j] + radii[k] - dist
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    force_vec = (diff / dist) * overlap * repulsion_strength
                    forces[j] += force_vec
                    forces[k] -= force_vec

        # Calculate boundary repulsion
        for j in range(n):
            x, y = centers[j]
            r = radii[j]
            
            # Left wall
            if x < r:
                forces[j, 0] += (r - x) * boundary_strength
            # Right wall
            if x > 1 - r:
                forces[j, 0] -= (x - (1 - r)) * boundary_strength
            # Bottom wall
            if y < r:
                forces[j, 1] += (r - y) * boundary_strength
            # Top wall
            if y > 1 - r:
                forces[j, 1] -= (y - (1 - r)) * boundary_strength

        # Update velocities and positions
        # Add some random noise based on temperature to escape local minima
        noise = np.random.normal(0, temp * 0.01, centers.shape)
        
        accelerations = forces + noise
        velocities = velocities * damping + accelerations * dt
        centers += velocities * dt

        # Expand radii uniformly
        radii += current_expansion
        
        # Clamp centers to stay reasonably inside (helps numerical stability)
        # But strict constraints are handled by forces
        centers = np.clip(centers, 0, 1)

    # 3. Post-processing: Solve for exact radii given final centers
    # The force-directed method gives a good arrangement of centers, 
    # but radii might be slightly inconsistent or suboptimal.
    # We can compute the maximum valid radius for each circle given the neighbors.
    # This is an iterative process because r_i depends on r_j.
    # However, a simple approximation is to take the minimum distance to any neighbor/boundary.
    
    # Re-optimize radii using a simple iterative shrink/grow
    # Start with the radii from simulation
    radii_final = radii.copy()
    
    for _ in range(100):
        changed = False
        for j in range(n):
            max_r = min(centers[j, 0], 1 - centers[j, 0], 
                        centers[j, 1], 1 - centers[j, 1])
            for k in range(n):
                if j == k: continue
                dist = np.linalg.norm(centers[j] - centers[k])
                # r_j + r_k <= dist  =>  r_j <= dist - r_k
                candidate = dist - radii_final[k]
                if candidate < max_r:
                    max_r = candidate
            
            if max_r < radii_final[j] - 1e-9:
                radii_final[j] = max(max_r, 0)
                changed = True
            elif max_r > radii_final[j] + 1e-9:
                # Try to grow if possible
                # But we must ensure we don't violate constraints with others who are at their max
                # Simple greedy growth
                radii_final[j] = max_r
                changed = True
        if not changed:
            break

    # Final check and cleanup
    # Ensure strict validity
    for j in range(n):
        r = radii_final[j]
        # Boundary checks
        if r > centers[j, 0]: radii_final[j] = centers[j, 0]
        if r > 1 - centers[j, 0]: radii_final[j] = 1 - centers[j, 0]
        if r > centers[j, 1]: radii_final[j] = centers[j, 1]
        if r > 1 - centers[j, 1]: radii_final[j] = 1 - centers[j, 1]

    # Overlap checks - shrink if necessary
    for j in range(n):
        for k in range(j + 1, n):
            dist = np.linalg.norm(centers[j] - centers[k])
            sum_r = radii_final[j] + radii_final[k]
            if sum_r > dist:
                overlap = sum_r - dist
                # Shrink both equally
                radii_final[j] -= overlap / 2
                radii_final[k] -= overlap / 2
                # Ensure non-negative
                radii_final[j] = max(0, radii_final[j])
                radii_final[k] = max(0, radii_final[k])

    sum_radii = np.sum(radii_final)
    return centers, radii_final, sum_radii

# sol_000165 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3daa574a) state=331d9d05 sum of radii=2.378958 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

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

def run_packing():
    np.random.seed(42)
    N = 26
    
    # 1. Initialization: 5x5 grid + 1 center point
    centers = []
    # Grid for 25 circles
    grid_size = 5
    step = 1.0 / (grid_size + 1)
    for r in range(1, grid_size + 1):
        for c in range(1, grid_size + 1):
            centers.append([c * step, r * step])
    # Add 26th circle in the center
    centers.append([0.5, 0.5])
    centers = np.array(centers)
    
    radii = np.zeros(N)
    # Initial small radius
    radii[:] = 0.01

    # 2. Force-Directed Layout with Radius Expansion
    num_iters = 2000
    # Physics parameters
    k_repulsion = 100.0  # Spring constant for overlap
    k_wall = 500.0       # Strong force from walls
    k_expand = 0.05      # Force pushing radii to grow
    damping = 0.8
    max_radius = 0.5     # Physical limit

    # Velocity for centers
    velocities = np.zeros_like(centers)

    for step in range(num_iters):
        forces = np.zeros_like(centers)
        radii_forces = np.zeros(N)

        # Current radii growth
        # We gradually increase radii. 
        # A simple strategy: apply a constant positive force to radii.
        radii_forces[:] = k_expand

        for i in range(N):
            # Wall repulsion
            # Left wall
            if centers[i, 0] - radii[i] < 0:
                dist_wall = centers[i, 0] - radii[i]
                forces[i, 0] -= k_wall * (-dist_wall) # Push right
                radii_forces[i] -= k_wall * 1.0 # Penalize radius growth if touching wall
            # Right wall
            if centers[i, 0] + radii[i] > 1:
                dist_wall = centers[i, 0] + radii[i] - 1
                forces[i, 0] -= k_wall * dist_wall # Push left
                radii_forces[i] -= k_wall * 1.0
            # Bottom wall
            if centers[i, 1] - radii[i] < 0:
                dist_wall = centers[i, 1] - radii[i]
                forces[i, 1] -= k_wall * (-dist_wall) # Push up
                radii_forces[i] -= k_wall * 1.0
            # Top wall
            if centers[i, 1] + radii[i] > 1:
                dist_wall = centers[i, 1] + radii[i] - 1
                forces[i, 1] -= k_wall * dist_wall # Push down
                radii_forces[i] -= k_wall * 1.0

            # Inter-circle repulsion
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist == 0:
                    dist = 1e-9
                    diff = np.random.rand(2) # Random push if coincident
                
                req_dist = radii[i] + radii[j]
                if dist < req_dist:
                    overlap = req_dist - dist
                    # Force proportional to overlap
                    force_mag = k_repulsion * overlap
                    direction = diff / dist
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
                    
                    # If overlapping, penalize radius growth? 
                    # Actually, just moving centers apart is usually enough.
                    # But to prevent infinite growth, we can reduce growth rate if stuck?
                    # For now, rely on center movement.

        # Update velocities and positions
        velocities += forces * 0.01 # dt
        velocities *= damping
        centers += velocities * 0.01

        # Update radii
        radii += radii_forces * 0.001
        radii = np.clip(radii, 0, max_radius)

        # Clamp centers to bounds [0, 1] loosely, forces handle the rest
        # But hard clamp prevents flying away
        centers = np.clip(centers, 1e-9, 1.0 - 1e-9)

    # 3. Refinement: Local Search to tighten packing
    # Try to increase radii locally
    for _ in range(500):
        # Pick random circle
        idx = np.random.randint(0, N)
        
        # Try to increase radius
        current_r = radii[idx]
        # Calculate max possible radius for this circle given others
        max_r = 1.0 # Upper bound
        
        # Check walls
        max_r = min(max_r, centers[idx, 0], 1 - centers[idx, 0], centers[idx, 1], 1 - centers[idx, 1])
        
        # Check neighbors
        for j in range(N):
            if idx == j: continue
            dist = np.linalg.norm(centers[idx] - centers[j])
            possible_r = dist - radii[j]
            if possible_r < max_r:
                max_r = possible_r
        
        if max_r > current_r:
            radii[idx] = max_r - 1e-9 # Leave tiny buffer

    # 4. Final Cleanup: Ensure strict validity
    # Run a quick relaxation to fix any tiny overlaps from the local search
    for _ in range(100):
        for i in range(N):
            # Resolve overlaps with neighbors
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if dist < req:
                    # Push apart equally
                    overlap = req - dist
                    if dist > 0:
                        shift = (centers[i] - centers[j]) / dist * (overlap / 2)
                        centers[i] += shift
                        centers[j] -= shift
                    # Clamp back to bounds if necessary
                    centers[i] = np.clip(centers[i], radii[i], 1 - radii[i])
                    centers[j] = np.clip(centers[j], radii[j], 1 - radii[j])
            
            # Resolve wall overlaps
            # Left
            if centers[i, 0] < radii[i]:
                centers[i, 0] = radii[i]
            # Right
            if centers[i, 0] > 1 - radii[i]:
                centers[i, 0] = 1 - radii[i]
            # Bottom
            if centers[i, 1] < radii[i]:
                centers[i, 1] = radii[i]
            # Top
            if centers[i, 1] > 1 - radii[i]:
                centers[i, 1] = 1 - radii[i]

    # Calculate sum
    sum_radii = np.sum(radii)
    
    # Final validation check (debug)
    # if not validate_packing(centers, radii):
    #     print("WARNING: Final packing invalid")
    
    return centers, radii, sum_radii

# Execute and print result for verification
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Min radius: {np.min(r)}, Max radius: {np.max(r)}")
    # Run validation from prompt
    print(f"Valid: {validate_packing(c, r)}")

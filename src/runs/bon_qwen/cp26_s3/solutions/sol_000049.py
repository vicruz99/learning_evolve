# sol_000049 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state baeb2167) state=daf606e5 sum of radii=1.992943 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    centers = np.zeros((n_circles, 2))
    
    # 1. Initialize centers in a hexagonal-like grid pattern
    # We aim for a layout that approximates hexagonal packing density.
    # 26 circles: try 5 rows. 6, 5, 6, 5, 4 or similar distribution.
    # Let's try to fit them in a rectangle and scale.
    
    # Generate points on a hexagonal lattice
    points = []
    # Approximate rows needed: sqrt(26 * 2 / sqrt(3)) ~ 5.5 rows. Let's use 6 rows.
    rows = 6
    cols = 5 # 6*5 = 30 points, we will pick 26
    
    # Hexagonal lattice parameters
    dx = 2.0  # horizontal spacing (relative)
    dy = math.sqrt(3)  # vertical spacing (relative)
    
    y = 0
    for r in range(rows):
        x = 0
        # Offset every other row
        if r % 2 == 1:
            x = dx / 2.0
        
        # Determine number of columns for this row to get close to 26
        # Total needed 26. 6 rows. Average 4.33 per row.
        # Let's vary columns: 5, 4, 5, 4, 5, 3 -> sum 26? 5+4+5+4+5+3 = 26.
        # Or 5, 5, 5, 5, 4, 2?
        # Let's just generate a grid and pick the first 26.
        for c in range(cols + 1):
            points.append([x, y])
            x += dx
        y += dy
    
    # Take first 26 points
    selected_points = points[:n_circles]
    
    # Normalize coordinates to fit in [0,1] with some padding
    xs = [p[0] for p in selected_points]
    ys = [p[1] for p in selected_points]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    if range_x == 0: range_x = 1
    if range_y == 0: range_y = 1
    
    padding = 0.15 # Leave room for radii expansion
    
    for i, p in enumerate(selected_points):
        # Normalize to [padding, 1-padding]
        nx = padding + (p[0] - min_x) / range_x * (1 - 2*padding)
        ny = padding + (p[1] - min_y) / range_y * (1 - 2*padding)
        centers[i] = [nx, ny]

    # Initial small radius
    current_r = 0.02
    radii = np.full(n_circles, current_r)

    # 2. Force-directed optimization
    # Parameters
    num_iterations = 2000
    expansion_rate = 0.00015 # How fast to grow radii
    repulsion_strength = 2.0
    boundary_strength = 5.0
    damping = 0.5
    
    velocities = np.zeros((n_circles, 2))

    for step in range(num_iterations):
        forces = np.zeros((n_circles, 2))
        
        # Calculate pairwise repulsion forces
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    # Force magnitude
                    f_mag = repulsion_strength * overlap / (dist + 1e-9)
                    f_vec = f_mag * diff / dist # Normalize diff to get direction
                    forces[i] += f_vec
                    forces[j] -= f_vec

        # Calculate boundary forces
        for i in range(n_circles):
            r = radii[i]
            x, y = centers[i]
            
            # Left wall
            if x - r < 0:
                push = boundary_strength * (r - x)
                forces[i, 0] += push
            # Right wall
            if x + r > 1:
                push = boundary_strength * (x + r - 1)
                forces[i, 0] -= push
            # Bottom wall
            if y - r < 0:
                push = boundary_strength * (r - y)
                forces[i, 1] += push
            # Top wall
            if y + r > 1:
                push = boundary_strength * (y + r - 1)
                forces[i, 1] -= push

        # Update velocities and positions
        # Velocity Verlet-like integration or simple Euler with damping
        velocities = velocities * damping + forces
        centers += velocities * 0.05 # Step size scaling

        # Clamp centers to [0, 1] to prevent numerical escape
        np.clip(centers, 0, 1, out=centers)

        # Gradually increase radius
        # We check if the system is stable (forces small) before increasing much?
        # Or just steadily increase.
        # To be safe, only increase if max force is low?
        # Actually, steady growth works well with repulsion.
        current_r += expansion_rate
        radii[:] = current_r

        # Occasionally shrink slightly if too much overlap detected to stabilize
        # Check max overlap
        max_overlap = 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j]:
                    overlap = (radii[i] + radii[j]) - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
        
        # If overlap is significant, shrink radii slightly to recover
        if max_overlap > 0.005:
            current_r -= 0.0005
            radii[:] = current_r

    # 3. Final Adjustment
    # Ensure strict validity by slightly shrinking radii if needed
    # Check overlaps and boundaries one last time and shrink
    
    # Safety margin
    margin = 1e-6
    
    # Calculate minimum separation / distance to wall to tighten radii
    min_clearance = 1.0
    
    for i in range(n_circles):
        # Boundary clearance
        x, y = centers[i]
        d_wall = min(x, 1-x, y, 1-y)
        if d_wall < min_clearance:
            min_clearance = d_wall
            
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            # If dist < 2*r, clearance is negative?
            # We want to find r such that 2r <= dist and r <= d_wall
            # So r <= dist/2 and r <= d_wall
            # Max possible r for this circle is min(dist/2 for all j, d_wall)
            # But radii must be equal in our current logic? 
            # The problem allows different radii.
            # But our simulation assumed equal radii for simplicity of growth.
            # Let's try to set radii to the maximum possible equal radius found.
            pass

    # The simulation converged to some current_r.
    # Let's verify and tighten.
    final_r = current_r
    
    # Check constraints with final_r
    # If any violation, reduce r
    valid = True
    r_reduction = 0
    
    for i in range(n_circles):
        x, y = centers[i]
        if x - final_r < -1e-12 or x + final_r > 1 + 1e-12 or \
           y - final_r < -1e-12 or y + final_r > 1 + 1e-12:
            # Calculate needed reduction
            needed = max(0, -(x - final_r), (x + final_r) - 1, -(y - final_r), (y + final_r) - 1)
            r_reduction = max(r_reduction, needed)
            
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < 2 * final_r - 1e-12:
                needed = (2 * final_r - dist) / 2.0
                r_reduction = max(r_reduction, needed)

    if r_reduction > 0:
        final_r -= r_reduction + 1e-6 # Add small buffer

    radii[:] = final_r
    
    # Double check with validator logic (mental check)
    # Just to be super safe, enforce constraints by clamping radii individually?
    # The problem asks to maximize sum of radii.
    # If we have valid equal radii, that's a valid solution.
    # But maybe we can increase some radii?
    # With 26 circles, equal radii is usually near optimal for sum.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper to ensure the function is callable and correct structure
# The problem requires run_packing to be defined.

if __name__ == "__main__":
    # Self-test
    c, r, s = run_packing()
    # Basic print
    print(f"Sum of radii: {s}")
    print(f"Min radius: {np.min(r)}, Max radius: {np.max(r)}")
    # Validate overlaps
    n = len(c)
    ok = True
    for i in range(n):
        x, y = c[i]
        rad = r[i]
        if x-rad < -1e-9 or x+rad > 1+1e-9 or y-rad < -1e-9 or y+rad > 1+1e-9:
            print(f"Circle {i} out of bounds")
            ok = False
        for j in range(i+1, n):
            dist = np.sqrt((c[i][0]-c[j][0])**2 + (c[i][1]-c[j][1])**2)
            if dist < r[i] + r[j] - 1e-9:
                print(f"Overlap {i}-{j}")
                ok = False
    if ok:
        print("Packing is valid.")
    else:
        print("Packing is INVALID.")

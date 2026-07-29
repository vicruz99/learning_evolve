# sol_000223 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 76d635d8) state=ed7d25c9 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Strategy: 
    # 1. Initialize 26 circles in a hexagonal lattice configuration.
    # 2. Use a simple numerical optimization (minimize negative sum of radii)
    #    with constraints to keep circles within bounds and non-overlapping.
    #    However, solving 78 variables with 300+ constraints is slow/heavy.
    #    Instead, we use a heuristic "repulsive force" simulation to find a high-density packing.
    #    Then we calculate the optimal radii for that fixed configuration.

    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # --- Step 1: Initialization ---
    # Pack circles in a hexagonal pattern
    idx = 0
    r_init = 0.09  # Initial guess
    row_spacing = r_init * np.sqrt(3)
    
    rows = []
    # Try to fit rows
    y = r_init
    while y < 1.0 - r_init:
        row = []
        x = r_init
        # Check if we need offset for hexagonal packing
        if len(rows) % 2 == 1:
            x += r_init
        while x < 1.0 - r_init and idx < n:
            row.append([x, y])
            x += 2 * r_init
            idx += 1
        if row:
            rows.append(row)
        y += row_spacing

    # If we didn't fit all, fill remaining in grid
    if idx < n:
        # Simple grid fill for remainder
        grid_step = 0.1
        y = 0.05
        while idx < n:
            x = 0.05
            while idx < n:
                centers[idx] = [x, y]
                idx += 1
                x += grid_step
                if x > 0.95: break
            y += grid_step

    if idx > n:
        centers = centers[:n]

    # --- Step 2: Optimization via Simulated Annealing / Gradient Ascent ---
    # We want to maximize sum(r_i). 
    # r_i is determined by min distance to boundary and min distance to other centers / 2.
    # Actually, if we fix centers, the optimal radii are constrained by neighbors.
    # But a simpler objective for a fixed number of circles is to maximize the MINIMUM radius (equal circles packing).
    # However, the problem asks to maximize SUM of radii.
    # For dense packings, equal radii is a good approximation.
    # Let's try to optimize the positions to maximize the minimum separation (which correlates with max radius).
    
    # Objective: Maximize min_dist_to_boundary and min_dist_between_centers.
    # Let's scale the configuration. 
    # If we find positions for unit circles (diameter 2), the scaling factor S will determine the radius.
    
    # Let's use a force-directed layout to spread points.
    positions = centers.copy()
    
    # Repulsion strength
    k_repulse = 1.0
    # Attraction to center to keep them in box? No, boundaries are hard constraints.
    
    lr = 0.01
    temp = 1.0
    
    # We will iterate and adjust positions to increase the "slack"
    # Slack = min( dist_to_boundary, dist_to_neighbor/2 )
    # Actually, let's just maximize the minimum distance between points, assuming they stay in [0,1]^2.
    # The radius will be min(min_dist_points/2, min_dist_boundary).
    
    for step in range(2000):
        grad = np.zeros_like(positions)
        
        # Calculate forces
        for i in range(n):
            # Boundary forces
            x, y = positions[i]
            # If x < 0.01, push right. If x > 0.99, push left.
            if x < 0.01:
                grad[i, 0] += (0.01 - x) * 100
            elif x > 0.99:
                grad[i, 0] -= (x - 0.99) * 100
                
            if y < 0.01:
                grad[i, 1] += (0.01 - y) * 100
            elif y > 0.99:
                grad[i, 1] -= (y - 0.99) * 100
                
            # Pairwise repulsion
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 0.0001:
                    dist = 0.0001
                    diff = np.random.rand(2) * 0.001 # Random push
                
                # We want distance to be at least some target. 
                # Let's set target based on current density.
                # Or just repulsive force 1/dist^2
                force = k_repulse / (dist ** 2)
                dir_vec = diff / dist
                grad[i] += dir_vec * force
                grad[j] -= dir_vec * force

        # Update positions
        positions += lr * grad
        
        # Decay learning rate and increase repulsion to push harder
        if step % 100 == 0:
            lr *= 0.9
            k_repulse *= 1.1
        
        # Ensure bounds
        positions = np.clip(positions, 0, 1)

    # --- Step 3: Calculate Optimal Radii for Fixed Centers ---
    # For a fixed set of centers, the problem of maximizing sum of radii is an LP.
    # Maximize sum(r_i)
    # Subject to: r_i + r_j <= dist(i, j)
    #             0 <= r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # However, solving LP inside is slow. We can approximate.
    # For dense packings, r_i is roughly determined by the closest neighbor.
    # r_i approx min( boundary_limit_i, min_j( dist(i,j)/2 ) )?
    # Actually, if r_i = dist(i,j)/2 and r_j = dist(i,j)/2, then r_i+r_j = dist(i,j).
    # This satisfies the constraint.
    # Is it possible to have r_i + r_j > dist(i,j)? No.
    # Is it possible to have larger radii?
    # If we set r_i = min( boundary_limit_i, min_j( dist(i,j)/2 ) ), we might be conservative.
    # Example: 3 circles in line 1-2-3. d12=1, d23=1.
    # Naive: r1=0.5, r2=0.5, r3=0.5. Sum=1.5.
    # But r2 is constrained by both. r1+r2<=1, r2+r3<=1.
    # If r2=0.5, r1<=0.5, r3<=0.5.
    # What if r2=0? r1<=1, r3<=1. Sum=2.
    # But boundary limits usually cap r at 0.5.
    # With boundary limits, the naive assignment is often optimal or very close.
    # Let's use the naive assignment first, then maybe a quick local adjustment.
    
    max_radii = np.zeros(n)
    for i in range(n):
        x, y = positions[i]
        dist_boundary = min(x, 1-x, y, 1-y)
        min_dist_neighbor = np.inf
        for j in range(n):
            if i == j: continue
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist_neighbor:
                min_dist_neighbor = d
        
        # The radius is limited by boundary and by neighbors.
        # If we assume all neighbors have radius r, then 2r <= d => r <= d/2.
        # So r_i <= min_dist_neighbor / 2.
        max_radii[i] = min(dist_boundary, min_dist_neighbor / 2)

    # Refine radii using a small optimization or iterative adjustment
    # Since sum of radii is linear, we can try to shift radii from "congested" circles to "free" circles.
    # But for simplicity and robustness, the naive assignment is a valid packing.
    # Let's verify validity and return.
    
    centers = positions
    radii = max_radii
    
    # Basic validation
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            radii[i] = min(x, 1-x, y, 1-y) # Clamp to boundary
            valid = False
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j] - 1e-12:
                # Overlap! Reduce radii
                # Simple fix: scale down both radii proportionally to fit distance
                sum_r = radii[i] + radii[j]
                if sum_r > 0:
                    scale = d / sum_r
                    radii[i] *= scale
                    radii[j] *= scale
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

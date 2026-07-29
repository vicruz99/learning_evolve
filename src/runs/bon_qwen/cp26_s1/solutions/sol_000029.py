# sol_000029 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b079e3ed) state=be3d96eb sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a growing bubbles simulation with repulsion forces.
    """
    n = 26
    
    # Initialization: Hexagonal grid to start with a dense packing
    np.random.seed(42)
    r_init = 0.08
    centers = []
    y = r_init
    row = 0
    # Generate hexagonal points
    while y < 1.0 - r_init:
        x = r_init
        if row % 2 == 1:
            x += r_init
        while x < 1.0 - r_init:
            if len(centers) < n:
                centers.append([x, y])
            x += 2 * r_init
        y += r_init * math.sqrt(3)
        row += 1
    
    # Fill remaining spots if any (should not happen with r=0.08, but safety measure)
    while len(centers) < n:
        centers.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)])
        
    centers = np.array(centers[:n])
    radii = np.full(n, r_init)
    
    # Optimization loop: Grow radii and resolve overlaps
    # We balance growth rate and relaxation steps to find a valid high-radius packing
    growth_rate = 0.0015
    sim_steps = 30
    num_growth_steps = 80
    
    for step in range(num_growth_steps):
        radii += growth_rate
        
        # Relaxation steps to resolve overlaps for the new radii
        for _ in range(sim_steps):
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            for i in range(n):
                xi, yi = centers[i]
                ri = radii[i]
                
                # Wall repulsion forces
                # Left wall
                if xi < ri:
                    fx[i] += (ri - xi) * 50.0
                # Right wall
                if xi > 1.0 - ri:
                    fx[i] -= (xi - (1.0 - ri)) * 50.0
                # Bottom wall
                if yi < ri:
                    fy[i] += (ri - yi) * 50.0
                # Top wall
                if yi > 1.0 - ri:
                    fy[i] -= (yi - (1.0 - ri)) * 50.0
                
                # Pairwise repulsion forces
                for j in range(i + 1, n):
                    xj, yj = centers[j]
                    rj = radii[j]
                    
                    dx = xi - xj
                    dy = yi - yj
                    d_sq = dx*dx + dy*dy
                    d = math.sqrt(d_sq)
                    
                    min_dist = ri + rj
                    
                    if d < min_dist:
                        if d < 1e-9:
                            # Coincident centers, push apart randomly to break symmetry
                            angle = np.random.uniform(0, 2 * math.pi)
                            force = 10.0
                            fx[i] += math.cos(angle) * force
                            fy[i] += math.sin(angle) * force
                            fx[j] -= math.cos(angle) * force
                            fy[j] -= math.sin(angle) * force
                        else:
                            # Repulsion force proportional to overlap
                            overlap = min_dist - d
                            f_mag = overlap * 20.0 # Stiffness
                            nx = dx / d
                            ny = dy / d
                            fx[i] += nx * f_mag
                            fy[i] += ny * f_mag
                            fx[j] -= nx * f_mag
                            fy[j] -= ny * f_mag
            
            # Update centers based on forces
            # Using a simple Euler step with a time constant
            centers[:, 0] += fx * 0.05
            centers[:, 1] += fy * 0.05
            
            # Strictly enforce boundaries to prevent floating point drift
            for i in range(n):
                r = radii[i]
                centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
                centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)

    # Final repair: Scale down radii uniformly if overlaps still exist
    # This ensures the returned packing is strictly valid.
    s = 1.0
    for i in range(n):
        ri = radii[i]
        # Check wall constraints
        if ri > 0:
            s = min(s, centers[i, 0] / ri)
            s = min(s, (1.0 - centers[i, 0]) / ri)
            s = min(s, centers[i, 1] / ri)
            s = min(s, (1.0 - centers[i, 1]) / ri)
        
        # Check pairwise constraints
        for j in range(i + 1, n):
            rj = radii[j]
            sum_r = ri + rj
            if sum_r > 0:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = math.sqrt(dx*dx + dy*dy)
                s = min(s, d / sum_r)
    
    # Apply scale with a small margin to be safe against numerical errors
    s = max(0.0, s * 0.999)
    radii *= s
    
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii

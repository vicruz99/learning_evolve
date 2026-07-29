# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 556f0961) state=5bdfb804 sum of radii=2.227173 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    np.random.seed(42)
    n = 26

    # --- Phase 1: Initialization ---
    # Create a hexagonal grid. 
    # A hexagonal grid is denser than a square grid.
    # We generate more points than needed and trim them.
    points = []
    r_init = 0.1
    # Hexagonal packing geometry
    # Horizontal spacing: 2r, Vertical spacing: sqrt(3)r
    # Offset alternate rows by r
    
    cols = 8
    rows = 8
    
    # We generate a grid of potential centers
    potential_centers = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            x = c_idx * 2 * r_init + (r_idx % 2) * r_init
            y = r_idx * math.sqrt(3) * r_init
            potential_centers.append([x, y])
            
    # Filter points that are roughly inside the square to select best candidates
    # We want points that are inside [0, 1] x [0, 1]
    valid_centers = []
    for p in potential_centers:
        if 0 <= p[0] <= 1 and 0 <= p[1] <= 1:
            valid_centers.append(p)
            
    # If we don't have enough valid points, add random ones
    if len(valid_centers) < n:
        while len(valid_centers) < n:
            valid_centers.append([np.random.rand(), np.random.rand()])
            
    # Select n points. To maximize sum of radii, we prefer points that are 
    # well-distributed. Let's just take the first n valid points.
    initial_centers = np.array(valid_centers[:n])
    
    # To get a better start, let's ensure they are somewhat centered 
    # if the grid was too skewed, but the hex grid above is decent.
    # Let's add a bit of noise to break symmetry
    initial_centers += np.random.normal(0, 0.01, initial_centers.shape)
    
    # Initial radii: small, to ensure validity
    radii = np.full(n, 0.02)

    # --- Phase 2: Optimization ---
    centers = initial_centers
    radii = np.full(n, 0.02)
    
    # Function to calculate max possible radius for a circle given others
    def get_max_radius(i, centers, radii):
        x, y = centers[i]
        r = radii[i]
        
        # Boundary constraints
        max_r = min(x, 1 - x, y, 1 - y)
        
        # Overlap constraints with other circles
        for j in range(n):
            if i == j:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            # Constraint: r_i + r_j <= dist  =>  r_i <= dist - r_j
            limit = dist - radii[j]
            if limit < max_r:
                max_r = limit
                
        return max(0, max_r)

    # Optimization Loop
    # We run multiple cycles of expansion and perturbation
    for iteration in range(50):
        # 1. Expand Radii
        # Iteratively expand until convergence
        for _ in range(10):
            improved = False
            for i in range(n):
                new_r = get_max_radius(i, centers, radii)
                if new_r > radii[i] + 1e-9:
                    radii[i] = new_r
                    improved = True
            if not improved:
                break
        
        # 2. Perturb Centers (Jiggle)
        # Try moving each circle to see if it allows a global increase in sum of radii
        # We do this by checking if moving circle i allows it (and neighbors) to grow
        step_size = 0.05 * (1.0 - iteration / 100.0) # Decrease step size over time
        
        for i in range(n):
            # Try several random directions
            for _ in range(5):
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                
                old_x, old_y = centers[i]
                centers[i, 0] = np.clip(old_x + dx, 0, 1)
                centers[i, 1] = np.clip(old_y + dy, 0, 1)
                
                # Calculate new sum of radii if we optimize radii after move
                # To save time, we just check if circle i can grow significantly
                # or if it doesn't overlap too much
                
                # A simpler heuristic: check if the move is valid and potentially beneficial
                # We do a quick local radius expansion for circle i
                # But we must respect current radii of others
                
                # Check validity of move with CURRENT radii
                valid_move = True
                # Boundary check is handled by clip, but let's be sure
                if centers[i, 0] - radii[i] < 0 or centers[i, 0] + radii[i] > 1 or \
                   centers[i, 1] - radii[i] < 0 or centers[i, 1] + radii[i] > 1:
                    valid_move = False
                
                if valid_move:
                    for j in range(n):
                        if i == j: continue
                        dist = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                        if dist < radii[i] + radii[j] - 1e-12:
                            valid_move = False
                            break
                
                if valid_move:
                    # If move is valid, try to expand circle i
                    new_r = get_max_radius(i, centers, radii)
                    # If we can expand, or if we are just shuffling for better fit
                    # We accept if sum of radii increases (approximate)
                    # Or simply accept valid moves that don't shrink others too much
                    
                    # Let's use a "grow" strategy: if we can grow, accept.
                    if new_r > radii[i]:
                        radii[i] = new_r
                    else:
                        # Even if we can't grow immediately, the move might be better for neighbors
                        # But to keep it simple and stable, let's stick to growth or random walk
                        # Let's revert if no immediate gain and it's a random walk
                        # Actually, let's just keep the move if valid, it might open up space
                        # But we need to be careful not to create overlaps for others
                        # Since we checked validity against current radii, it's safe
                        pass 
                else:
                    # Revert
                    centers[i, 0] = old_x
                    centers[i, 1] = old_y
                    # If move was invalid, maybe try a smaller step? 
                    # But we loop multiple times so it's fine.

    # Final cleanup: Ensure validity and maximize radii one last time
    for _ in range(50):
        for i in range(n):
            radii[i] = get_max_radius(i, centers, radii)
            
    # Sort radii for deterministic output (optional but good for checking)
    # Actually, the problem doesn't require sorted radii, just arrays.
    
    sum_radii = np.sum(radii)
    
    # Validation check (internal)
    # Just to be safe, we can shrink slightly if any overlap occurs due to float errors
    # But get_max_radius should prevent this.
    
    return centers, radii, sum_radii

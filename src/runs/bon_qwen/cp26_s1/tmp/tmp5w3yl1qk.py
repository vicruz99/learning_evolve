import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Strategy:
    1. Initialize centers on a grid.
    2. Iteratively:
       a. Solve a Linear Program to maximize sum of radii for fixed centers.
       b. Apply repulsive forces to move centers apart, creating space for larger radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 1. Initialization: Grid placement
    # We need 26 points. A 6x5 grid gives 30 points.
    # We'll pick the first 26.
    rows = 5
    cols = 6
    x_vals = np.linspace(0.1, 0.9, cols) # Keep away from edges initially
    y_vals = np.linspace(0.1, 0.9, rows)
    
    grid_points = []
    for y in y_vals:
        for x in x_vals:
            grid_points.append([x, y])
            if len(grid_points) == n:
                break
        if len(grid_points) == n:
            break
            
    centers = np.array(grid_points)
    # Small initial radii to start valid
    radii[:] = 0.02 

    # 2. Optimization Loop
    n_iters = 500
    learning_rate = 0.05
    
    for step in range(n_iters):
        # --- Step A: Maximize Radii using LP ---
        # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
        # Variables: r_0, ..., r_25
        
        # c_obj for linprog (minimization)
        c_obj = -np.ones(n)
        
        # Bounds for r_i: [0, max_possible_by_walls]
        bounds = []
        for i in range(n):
            x, y = centers[i]
            # Distance to walls
            max_r_wall = min(x, 1-x, y, 1-y)
            bounds.append((0, max(max_r_wall, 1e-6))) # Ensure positive upper bound
            
        # Inequality constraints: r_i + r_j <= dist(i, j)
        # A_ub @ r <= b_ub
        # We only need constraints for pairs that are close or all pairs?
        # For correctness, all pairs. But for performance, maybe skip if dist is large?
        # With n=26, 26*25/2 = 325 constraints. Very fast.
        
        pairs = []
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                pairs.append((i, j))
                dists.append(d)
        
        if pairs:
            pairs = np.array(pairs)
            # Construct A_ub
            # A_ub will have shape (num_pairs, n)
            A_ub = np.zeros((len(pairs), n))
            for k, (i, j) in enumerate(pairs):
                A_ub[k, i] = 1.0
                A_ub[k, j] = 1.0
            
            b_ub = np.array(dists)
            
            # Solve LP
            # method='highs' is robust
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                radii = res.x
            else:
                # Fallback if LP fails (should not happen with valid bounds)
                pass
        else:
            pass

        # --- Step B: Relax Positions (Repulsion) ---
        # We want to move centers to increase distances, allowing radii to grow.
        # Force = sum of repulsion vectors.
        
        forces = np.zeros((n, 2))
        
        # Calculate pairwise forces
        # We want to push circles apart if they are "tight" (r_i + r_j approx dist)
        # Also general repulsion to prevent clumping.
        
        repulsion_strength = 0.1 / (step + 10) # Decay repulsion over time
        tightness_strength = 0.5
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    dist = 1e-9
                    diff = np.random.rand(2) * 0.01 # Jitter if identical
                
                unit_vec = diff / dist
                
                # Constraint check: r_i + r_j <= dist
                # If tight, push apart strongly.
                # If loose, push apart weakly (to explore).
                
                sum_r = radii[i] + radii[j]
                slack = dist - sum_r
                
                # Force magnitude
                # If slack < 0 (overlap), large push.
                # If slack ~ 0 (touching), push to create space.
                # If slack > 0, small push.
                
                # A heuristic force function
                if slack < 0.01:
                     force_mag = tightness_strength * (0.01 - slack) / (dist + 1e-9)
                else:
                     force_mag = repulsion_strength / (dist**2)
                
                # Apply force
                forces[i] += unit_vec * force_mag
                forces[j] -= unit_vec * force_mag

        # Wall repulsion (push away from boundaries)
        # If circle is touching wall (r ~ x), push inward.
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x < r + 0.01:
                forces[i, 0] += 0.1
            # Right wall
            if x > 1 - r - 0.01:
                forces[i, 0] -= 0.1
            # Bottom wall
            if y < r + 0.01:
                forces[i, 1] += 0.1
            # Top wall
            if y > 1 - r - 0.01:
                forces[i, 1] -= 0.1

        # Update centers
        centers += learning_rate * forces
        
        # Clip centers to valid range [0, 1]
        centers = np.clip(centers, 0, 1)
        
        # Safety: Ensure radii are valid for new centers (clip down if necessary)
        # This handles cases where center moved outside allowed radius zone
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, 1-x, y, 1-y)
            if radii[i] > max_r:
                radii[i] = max_r

    # Final validation check and clean up
    # Ensure no negative radii or NaN
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Validation helper included in prompt, but we can run a quick check here
if __name__ == "__main__":
    # Import numpy if not available in scope (though it is)
    import numpy as np
    
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    
    # Simple overlap check
    valid = True
    for i in range(26):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1-r or y < r or y > 1-r:
            valid = False
        for j in range(i+1, 26):
            d = np.sqrt((centers[i]-centers[j])**2).sum() # Wrong syntax for sum, fix below
            # Correct distance calc
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - 1e-9:
                valid = False
    print(f"Valid: {valid}")
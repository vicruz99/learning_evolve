# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 873af2c6) state=b589dcde sum of radii=0.355063 correctness=1.0
# stdout(first 200): Circles 5 and 16 overlap: dist=9.04012988076067e-06, r1+r2=1.0459791539572195e-05
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def get_initial_config(n=26, width=1.0, height=1.0):
    """
    Generate an initial valid configuration of n circles using a hexagonal lattice.
    """
    # Parameters for hexagonal packing
    # We want to fit points roughly in a grid. 
    # For 26 points, a 5x6 grid area is a good approximation.
    # Spacing s. 
    # Horizontal distance s, vertical distance s * sqrt(3)/2.
    
    # Let's try to generate more points than needed and pick the best ones, 
    # or just fill a rectangle.
    # Let's aim for a grid that covers the square.
    
    # Estimate spacing. Area per circle approx 1/26. 
    # r ~ sqrt(1/(26*pi)) ~ 0.11. d ~ 0.22.
    # Let's start with a spacing of 0.25 to be safe and valid.
    
    cols = 7
    rows = 5
    
    points = []
    s = 1.0 / cols  # Horizontal step
    h = s * np.sqrt(3) / 2 # Vertical step
    
    # Offset for odd/even rows to create hex lattice
    for r_idx in range(rows):
        for c_idx in range(cols):
            x = c_idx * s + (0.5 * s if r_idx % 2 == 1 else 0) + s/2 # Center in cell roughly
            y = r_idx * h + h/2
            
            if x < width and y < height:
                points.append([x, y])
    
    # If we don't have enough points, reduce spacing
    while len(points) < n:
        s *= 0.8
        h = s * np.sqrt(3) / 2
        points = []
        for r_idx in range(8): # Increase potential rows
            for c_idx in range(8):
                x = c_idx * s + (0.5 * s if r_idx % 2 == 1 else 0) + s/2
                y = r_idx * h + h/2
                if x < width and y < height:
                    points.append([x, y])
    
    points = np.array(points[:n])
    
    # Compute initial radii. 
    # Radius is half the minimum distance to any other point and boundaries.
    # We set a safety factor to ensure strict non-overlap initially.
    safety = 0.1
    radii = []
    for i in range(n):
        x, y = points[i]
        min_dist = 1.0 # Max possible distance
        
        # Dist to boundaries
        d_bound = min(x, 1-x, y, 1-y)
        min_dist = min(min_dist, d_bound)
        
        # Dist to other centers
        # Since we pick from a dense grid, checking all is O(N^2) but N=26 is small
        for j in range(n):
            if i == j: continue
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            min_dist = min(min_dist, dist)
            
        # Radius is half the distance to nearest neighbor/boundary
        r = (min_dist / 2.0) * safety
        radii.append(r)
        
    return points, np.array(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None
    
    # Run optimization multiple times with perturbations
    num_trials = 5
    
    for trial in range(num_trials):
        # 1. Initialization
        centers, radii = get_initial_config(n)
        
        # Add small random perturbation to centers to escape symmetries
        centers += np.random.uniform(-0.01, 0.01, centers.shape)
        # Ensure centers stay within bounds after perturbation
        centers = np.clip(centers, 0.01, 0.99)
        
        # Flatten variables for scipy: [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.concatenate([centers.flatten(), radii])
        
        # 2. Define Objective: Maximize sum of radii -> Minimize negative sum
        def objective(vars_flat):
            r_vec = vars_flat[2::3] # Radii are at indices 2, 5, 8...
            return -np.sum(r_vec)
        
        # 3. Define Constraints
        # Inequality constraints: g(x) >= 0
        def constraints(vars_flat):
            c_list = []
            
            # Extract arrays
            centers_opt = vars_flat[:2*n].reshape(n, 2)
            radii_opt = vars_flat[2*n:]
            
            # Boundary constraints
            # x - r >= 0  => x - r >= 0
            # 1 - x - r >= 0 => 1 - x - r >= 0
            for i in range(n):
                x, y = centers_opt[i]
                r = radii_opt[i]
                c_list.append(x - r)
                c_list.append(1.0 - x - r)
                c_list.append(y - r)
                c_list.append(1.0 - y - r)
            
            # Overlap constraints
            # dist^2 - (r_i + r_j)^2 >= 0
            for i in range(n):
                for j in range(i + 1, n):
                    dist_sq = np.sum((centers_opt[i] - centers_opt[j])**2)
                    r_sum_sq = (radii_opt[i] + radii_opt[j])**2
                    c_list.append(dist_sq - r_sum_sq)
            
            return np.array(c_list)

        # 4. Bounds
        # x, y in [0, 1]
        # r >= 0
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 1)]) # x, y, r
        
        # 5. Optimize
        # SLSQP is good for non-linear constrained optimization
        try:
            result = scipy.optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints},
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < -best_sum_radii + 0.01: # Check if improved
                # Extract results
                final_centers = result.x[:2*n].reshape(n, 2)
                final_radii = result.x[2*n:]
                
                current_sum = np.sum(final_radii)
                
                # Validate locally before updating best
                if validate_packing(final_centers, final_radii):
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = final_centers
                        best_radii = final_radii
        except Exception:
            continue

    # Fallback to initial config if optimization failed completely (unlikely)
    if best_centers is None:
        centers, radii = get_initial_config(n)
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    return best_centers, best_radii, best_sum_radii

# Execution block to run and print result
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    is_valid = validate_packing(centers, radii)
    print(f"Valid: {is_valid}")

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses an alternating optimization strategy:
    1. Solve LP for optimal radii given centers.
    2. Update centers using repulsive forces derived from tight constraints.
    """
    n = 26
    
    # --- 1. Initialization ---
    # Start with a 5x5 grid (25 points) plus one extra point.
    # This gives a good initial density.
    pts = []
    for i in range(5):
        for j in range(5):
            # Grid spacing 0.2, centered in cells? 
            # 0.1, 0.3, 0.5, 0.7, 0.9 are good centers for radius 0.1
            pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    
    # Add 26th circle. 
    # Try to place it in a gap. A perturbed center or specific location.
    # Placing at (0.5, 0.05) is close to bottom edge but might be tight.
    # Let's place it randomly near the center to let optimizer decide, 
    # or specifically at a hole if we knew one. 
    # A simple random jitter on a grid point might be safer.
    # Let's just add a point at (0.05, 0.55) - left side gap?
    # Actually, just random initialization works well with the optimizer.
    # But let's stick to a deterministic start for reproducibility.
    # 26 points. Maybe a hexagonal pattern is better?
    # Let's just add one at (0.5, 0.5) shifted slightly? 
    # (0.5, 0.5) is already in the grid (i=2, j=2).
    # Let's perturb the grid slightly to create space, or add point at corner.
    # Let's add at (0.5, 0.02) - very close to bottom.
    pts.append([0.5, 0.02])
    
    centers = np.array(pts)
    
    # --- 2. Setup LP Constraints ---
    # Variables: r_0 ... r_25
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    c_obj = np.ones(n) * -1
    
    # Constraints:
    # 1. Pairwise: r_i + r_j <= dist(c_i, c_j)
    # 2. Boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # 3. Non-negativity: r_i >= 0
    
    num_pairs = n * (n - 1) // 2
    num_boundary = 4 * n
    num_constraints = num_pairs + num_boundary
    
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    # Precompute pair indices
    pairs = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            A_ub[idx, i] = 1
            A_ub[idx, j] = 1
            idx += 1
            
    # Precompute boundary constraint rows
    # Rows from num_pairs to num_pairs + num_boundary
    base_idx = num_pairs
    for i in range(n):
        # r_i <= x_i
        A_ub[base_idx, i] = 1
        base_idx += 1
        # r_i <= 1-x_i
        A_ub[base_idx, i] = 1
        base_idx += 1
        # r_i <= y_i
        A_ub[base_idx, i] = 1
        base_idx += 1
        # r_i <= 1-y_i
        A_ub[base_idx, i] = 1
        base_idx += 1
        
    bounds = [(0, None)] * n
    
    # --- 3. Optimization Loop ---
    max_iter = 300
    step_size = 0.005  # Learning rate for center movement
    
    best_sum_r = 0.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    for iteration in range(max_iter):
        # Update RHS vector b_ub based on current centers
        
        # 1. Pairwise distances
        idx = 0
        for i, j in pairs:
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            b_ub[idx] = dist
            idx += 1
            
        # 2. Boundary distances
        idx = num_pairs
        for i in range(n):
            x, y = centers[i]
            b_ub[idx] = x;        idx += 1       # x_i
            b_ub[idx] = 1 - x;    idx += 1       # 1-x_i
            b_ub[idx] = y;        idx += 1       # y_i
            b_ub[idx] = 1 - y;    idx += 1       # 1-y_i
            
        # Solve LP
        # Using 'highs' method for speed and reliability
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
                current_sum = np.sum(radii)
                
                if current_sum > best_sum_r:
                    best_sum_r = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                
                # Compute repulsive forces to update centers
                forces = np.zeros((n, 2))
                
                # Check tight pairwise constraints
                # If r_i + r_j is very close to dist(i, j), push apart
                idx = 0
                for i, j in pairs:
                    # Slack = dist - (r_i + r_j)
                    # Using the updated b_ub[idx] which is current dist
                    slack = b_ub[idx] - (radii[i] + radii[j])
                    
                    if slack < 1e-4: # Tight constraint
                        vec = centers[i] - centers[j]
                        dist = np.linalg.norm(vec)
                        if dist > 1e-9:
                            dir = vec / dist
                            # Force magnitude proportional to radii sum to give larger circles more "influence"?
                            # Or just constant. Let's use a small constant factor.
                            # The step_size controls the magnitude of update.
                            f_mag = step_size * (radii[i] + radii[j]) * 0.5 
                            forces[i] += dir * f_mag
                            forces[j] -= dir * f_mag
                    idx += 1
                
                # Check tight boundary constraints
                idx = num_pairs
                for i in range(n):
                    x, y = centers[i]
                    r = radii[i]
                    
                    # r <= x (Left wall) -> If tight, push Right (+x)
                    if abs(r - x) < 1e-4:
                        forces[i, 0] += step_size * r
                    
                    # r <= 1-x (Right wall) -> If tight, push Left (-x)
                    if abs(r - (1 - x)) < 1e-4:
                        forces[i, 0] -= step_size * r
                        
                    # r <= y (Bottom wall) -> If tight, push Up (+y)
                    if abs(r - y) < 1e-4:
                        forces[i, 1] += step_size * r
                        
                    # r <= 1-y (Top wall) -> If tight, push Down (-y)
                    if abs(r - (1 - y)) < 1e-4:
                        forces[i, 1] -= step_size * r
                        
                    idx += 1 # Just consuming index, logic is per circle
                    
                    # Actually, the index increment in loop is for constraint rows.
                    # There are 4 rows per circle.
                    # My loop logic above skipped idx increment properly?
                    # Let's fix the index tracking.
                    # In the block above, I didn't increment idx correctly inside the if checks.
                    # But I don't need to check slack for boundaries via b_ub, I can check directly.
                    # So idx tracking is not needed here if I access centers/radii directly.
                
                # Apply forces
                centers += forces
                
                # Clip centers to stay within [0, 1] strictly? 
                # The LP handles boundaries, but moving centers might push them out if forces are large.
                # However, forces push AWAY from boundaries when tight.
                # If not tight, no force.
                # So centers should naturally stay inside. 
                # But numerical errors might occur.
                centers = np.clip(centers, 0, 1)
                
                # Add small random jitter to escape local minima?
                # Maybe occasionally.
                if iteration % 50 == 0 and iteration > 0:
                    centers += np.random.normal(0, 0.001, centers.shape)
                    centers = np.clip(centers, 0.001, 0.999) # Keep away from absolute edge
                    
        except Exception as e:
            # If LP fails, break or continue
            break

    # Final validation and return
    # Ensure best_radii are valid for best_centers?
    # The best_radii were computed for best_centers, so they are valid.
    # But let's double check sum.
    
    # Re-run one last LP to be sure radii are optimal for the best_centers found
    # (In case the loop ended on an update that moved centers but we didn't record the new radii yet)
    # Actually, we update best_radii inside the loop.
    # But after the loop, centers might have moved.
    # Let's re-solve for the final best_centers.
    
    # Update b_ub for best_centers
    idx = 0
    for i, j in pairs:
        dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
        b_ub[idx] = dist
        idx += 1
        
    idx = num_pairs
    for i in range(n):
        x, y = best_centers[i]
        b_ub[idx] = x;        idx += 1
        b_ub[idx] = 1 - x;    idx += 1
        b_ub[idx] = y;        idx += 1
        b_ub[idx] = 1 - y;    idx += 1
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            best_radii = res.x
            best_sum_r = np.sum(best_radii)
    except:
        pass

    return best_centers, best_radii, best_sum_r

if __name__ == "__main__":
    # Quick test
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    print(f"Centers shape: {centers.shape}")
    print(f"Radii shape: {radii.shape}")
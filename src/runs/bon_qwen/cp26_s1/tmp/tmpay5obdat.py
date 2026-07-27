import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # Function to compute the objective (sum of radii)
    # We want to maximize sum(r), so we minimize negative sum(r)
    def objective(vars):
        centers = vars[:2*n_circles].reshape(n_circles, 2)
        radii = vars[2*n_circles:]
        return -np.sum(radii)

    # Constraints:
    # 1. Non-overlap: dist(i, j) >= r_i + r_j
    #    dist^2 >= (r_i + r_j)^2
    #    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # 2. Boundary: 0 <= x - r, x + r <= 1, 0 <= y - r, y + r <= 1
    #    r <= x <= 1-r, r <= y <= 1-r
    #    This implies r <= 0.5, x >= r, 1-x >= r, etc.
    
    # We will use SLSQP which supports constraints.
    # However, with ~26*3 = 78 variables and ~26*26/2 + 26*4 constraints, it might be slow or fail.
    # A penalty method or a more heuristic approach might be better.
    # Let's try a hybrid: Generate a good initial guess, then refine.

    def generate_initial_guess():
        # Start with a 5x5 grid of 25 circles, r=0.1
        # Then perturb and add the 26th.
        # Or better: Hexagonal packing.
        
        # Let's try a dense random perturbation of a grid
        centers = np.zeros((n_circles, 2))
        radii = np.full(n_circles, 0.095) # Slightly less than 0.1 to allow movement
        
        # Place first 25 in a 5x5 grid
        for i in range(25):
            row = i // 5
            col = i % 5
            centers[i, 0] = 0.1 + col * 0.2
            centers[i, 1] = 0.1 + row * 0.2
            
        # Place 26th in a gap, e.g., center of a cell?
        # Or just somewhere random valid
        centers[25, 0] = 0.5
        centers[25, 1] = 0.5
        # This will overlap heavily, so optimization needs to resolve it.
        # Better initial placement:
        # Try to place 26th circle in a hole. In 5x5 grid, no holes.
        # So we must shrink all.
        # Let's use a hexagonal packing pattern for better density
        # Hex packing of 26 circles?
        
        # Alternative: Random restarts with local search
        return centers, radii

    # We will use a simple iterative expansion approach
    # 1. Place circles randomly or in a pattern with small radii.
    # 2. Expand radii until constraints hit.
    # 3. Move centers to free up space.
    
    # Given the complexity, let's use a standard optimizer with a good start.
    # Start with 5x5 grid, r=0.09 (safe margin), and 26th circle small.
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Define constraints for SLSQP
    # Inequality constraints: g(x) >= 0
    
    constraints = []
    
    # Boundary constraints
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    # We can encode these in bounds or explicit constraints.
    # Bounds for x, y: [0, 1]. Bounds for r: [0, 0.5].
    # But r depends on x,y.
    # Let's use explicit constraints for safety or bounds.
    # Actually, if we set bounds for x in [0,1], r in [0, 0.5], 
    # we still need x-r>=0 etc.
    
    # Let's construct constraints list
    for i in range(n_circles):
        idx_x = i * 2
        idx_y = i * 2 + 1
        idx_r = 2 * n_circles + i
        
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: vars[idx_x] - vars[idx_r]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: 1.0 - vars[idx_x] - vars[idx_r]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: vars[idx_y] - vars[idx_r]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: 1.0 - vars[idx_y] - vars[idx_r]
        })
        
    # Overlap constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            idx_xi = i * 2
            idx_yi = i * 2 + 1
            idx_ri = 2 * n_circles + i
            idx_xj = j * 2
            idx_yj = j * 2 + 1
            idx_rj = 2 * n_circles + j
            
            # dist^2 >= (ri + rj)^2
            # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            def make_overlap_con(i, j):
                def overlap(vars):
                    xi, yi, ri = vars[idx_xi], vars[idx_yi], vars[idx_ri]
                    xj, yj, rj = vars[idx_xj], vars[idx_yj], vars[idx_rj]
                    dx = xi - xj
                    dy = yi - yj
                    return dx*dx + dy*dy - (ri + rj)**2
                return overlap
            
            constraints.append({
                'type': 'ineq',
                'fun': make_overlap_con(i, j)
            })

    # Bounds
    # x, y in [0, 1], r in [0, 0.5] (actually r <= 0.5 is loose, r <= 0.1 is tighter but we don't know exact)
    bounds = []
    for i in range(n_circles):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)]) # x, y
        bounds.append((0.0, 0.5)) # r
    
    # Initial guess
    # 5x5 grid with r=0.09, and 26th circle at (0.5, 0.5) with r=0.01
    initial_centers = np.zeros((n_circles, 2))
    initial_radii = np.full(n_circles, 0.09)
    
    for i in range(25):
        row = i // 5
        col = i % 5
        initial_centers[i, 0] = 0.1 + col * 0.2
        initial_centers[i, 1] = 0.1 + row * 0.2
    
    initial_centers[25, 0] = 0.5
    initial_centers[25, 1] = 0.5
    initial_radii[25] = 0.001 # Start small for the new one
    
    x0 = np.concatenate([initial_centers.flatten(), initial_radii])
    
    # Run optimizer
    # SLSQP is good for constrained optimization
    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'ftol': 1e-9, 'maxiter': 1000})
        
        if res.success:
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = res.x[:2*n_circles].reshape(n_circles, 2)
                best_radii = res.x[2*n_circles:]
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Try a few random restarts to improve
    for _ in range(5):
        # Perturb the best found solution or generate new
        # Let's perturb best
        if best_centers is None:
            centers = initial_centers.copy()
            radii = initial_radii.copy()
        else:
            centers = best_centers.copy()
            radii = best_radii.copy()
            
        # Add noise
        noise = np.random.normal(0, 0.01, size=centers.shape)
        centers = np.clip(centers + noise, 0, 1)
        radii = np.clip(radii + np.random.normal(0, 0.005, size=n_circles), 0, 0.5)
        
        # Fix constraints if violated by noise (simple projection)
        for i in range(n_circles):
            r = radii[i]
            x, y = centers[i]
            if x < r: x = r
            if x > 1-r: x = 1-r
            if y < r: y = r
            if y > 1-r: y = 1-r
            centers[i] = [x, y]
            
        # Check overlaps and shrink if necessary (simple heuristic)
        # If overlap, reduce radii
        valid = False
        while not valid:
            valid = True
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                    if dist < radii[i] + radii[j]:
                        # Shrink both
                        reduction = (radii[i] + radii[j] - dist) / 2 + 1e-6
                        radii[i] -= reduction
                        radii[j] -= reduction
                        if radii[i] < 1e-4: radii[i] = 1e-4
                        if radii[j] < 1e-4: radii[j] = 1e-4
                        valid = False
            if not valid:
                # Re-check boundaries after shrinking? 
                # Shrinking radius makes it easier to satisfy boundary
                pass

        x0_pert = np.concatenate([centers.flatten(), radii])
        
        try:
            res = opt.minimize(objective, x0_pert, method='SLSQP', bounds=bounds, constraints=constraints, 
                               options={'ftol': 1e-9, 'maxiter': 500})
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[:2*n_circles].reshape(n_circles, 2)
                    best_radii = res.x[2*n_circles:]
        except:
            pass

    # Final validation and cleanup
    # Ensure radii are non-negative and valid
    if best_centers is None:
        best_centers = initial_centers
        best_radii = initial_radii
        best_sum = np.sum(best_radii)

    # Clip radii to be non-negative
    best_radii = np.maximum(best_radii, 1e-9)
    
    # Ensure centers are within bounds considering radius
    for i in range(n_circles):
        r = best_radii[i]
        x, y = best_centers[i]
        best_centers[i, 0] = np.clip(x, r, 1-r)
        best_centers[i, 1] = np.clip(y, r, 1-r)

    # Re-calculate sum
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum

# Execute to find the packing
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
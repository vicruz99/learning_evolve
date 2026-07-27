import numpy as np
from scipy.optimize import differential_evolution, linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Objective: Maximize sum of radii
    # We will use an iterative approach:
    # 1. Use Differential Evolution to find good centers.
    # 2. For fixed centers, solve LP to find max sum of radii.
    # However, DE optimizes a scalar function. We can define the function 
    # f(centers) = max_sum_radii(centers).
    # But linprog minimizes, so we minimize -sum(radii).
    
    # To speed up, we can fix radii to be equal during DE? 
    # No, radii should be optimized.
    # But calling linprog inside DE objective might be slow.
    # Alternative: Optimize centers and radii directly using a penalized objective or SLSQP.
    # Given the complexity and constraints, a simple randomized local search with 
    # inflation might be more robust and faster to implement without heavy dependencies.
    # But we have scipy.
    
    # Let's try a direct optimization using SLSQP with a good initial guess.
    # Initial guess: Hexagonal packing.
    
    def get_hexagonal_centers(n):
        centers = []
        # Try to fit n circles in a hexagonal pattern
        # Estimate number of rows
        # Area approx n * pi * r^2. If r ~ 0.1, area ~ 0.8.
        # Grid 5x5 is 25. Hexagonal 5x6 (staggered) is 30?
        # Let's try to arrange in rows.
        row_count = 5
        cols_per_row = []
        remaining = n
        for i in range(row_count):
            # Distribute columns
            c = n // row_count + (1 if i < n % row_count else 0)
            cols_per_row.append(c)
        
        # We need to determine spacing. Let's assume radius r=0.1 initially for layout
        r_est = 0.1
        dy = np.sqrt(3) * r_est # Vertical spacing
        dx = 2 * r_est         # Horizontal spacing
        
        # Center the grid in [0,1]x[0,1]
        # Calculate total height and width
        # Height: 2*r + (rows-1)*dy
        # Width varies per row.
        
        # Better: Generate points, then scale/translate to fit in square.
        raw_centers = []
        y = r_est
        for r_idx, count in enumerate(cols_per_row):
            x_start = r_est
            if r_idx % 2 == 1:
                x_start += r_est # Shift row
            for c_idx in range(count):
                raw_centers.append([x_start + c_idx * dx, y])
            y += dy
        
        raw_centers = np.array(raw_centers)
        
        # If we generated more than n, trim
        if len(raw_centers) > n:
            raw_centers = raw_centers[:n]
        
        # Normalize to fit in [0, 1]
        # Find bounding box
        min_x, min_y = np.min(raw_centers, axis=0)
        max_x, max_y = np.max(raw_centers, axis=0)
        
        # Scale to fit within [0, 1] with some padding? 
        # Actually, just scale to fill.
        width = max_x - min_x
        height = max_y - min_y
        
        scale_x = 1.0 / width
        scale_y = 1.0 / height
        scale = min(scale_x, scale_y)
        
        # To keep aspect ratio or just scale?
        # Scaling uniformly might not be optimal, but good start.
        # Let's just scale to fit.
        
        # Shift to 0
        raw_centers -= np.array([min_x, min_y])
        # Scale
        raw_centers *= scale
        
        # If we scaled based on min dimension, the other dimension fits.
        # But we might be far from boundaries.
        # Let's center it.
        # Actually, just return these centers.
        
        # Wait, if we have 26 circles, 5x5 grid is 25.
        # Hexagonal might need different counts.
        # 3, 5, 5, 5, 5, 3? Sum 26.
        # Let's use a robust distribution.
        
        # Re-calc with specific counts for better packing
        # Try 5 rows.
        # Counts: 5, 6, 5, 6, 4? Sum 26.
        counts = [5, 6, 5, 6, 4]
        # Or 5, 5, 5, 5, 6
        counts = [5, 5, 5, 5, 6]
        
        raw_centers = []
        y = 0
        for r_idx, count in enumerate(counts):
            x = 0
            if r_idx % 2 == 1:
                x += 1 # Shift
            for _ in range(count):
                raw_centers.append([x, y])
                x += 2
            y += np.sqrt(3)
        
        raw_centers = np.array(raw_centers)
        
        # Normalize
        min_c = np.min(raw_centers, axis=0)
        max_c = np.max(raw_centers, axis=0)
        
        # Scale to fit in [0,1]
        # We want to map [min, max] to [0, 1]
        # But we have 26 points.
        # Let's just return these normalized.
        
        scale_x = 1.0 / (max_c[0] - min_c[0])
        scale_y = 1.0 / (max_c[1] - min_c[1])
        
        # Use minimum scale to fit in square
        s = min(scale_x, scale_y)
        
        raw_centers -= min_c
        raw_centers *= s
        
        # Now it fits in [0, 1] roughly?
        # Max coord is 1.
        return raw_centers

    def solve_lp_radii(centers):
        # Maximize sum(r_i)
        # Subject to:
        # r_i >= 0
        # r_i <= x_i
        # r_i <= 1 - x_i
        # r_i <= y_i
        # r_i <= 1 - y_i
        # r_i + r_j <= dist(c_i, c_j)
        
        n = len(centers)
        # Variables: r_0, ..., r_{n-1}
        # Objective: minimize -sum(r) => c = -1
        
        c_obj = -np.ones(n)
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints: r_i <= bound
        # x - r >= 0 => -r >= -x => r <= x
        # x + r <= 1 => r <= 1 - x
        # y - r >= 0 => r <= y
        # y + r <= 1 => r <= 1 - y
        
        for i in range(n):
            x, y = centers[i]
            bounds_val = min(x, 1-x, y, 1-y)
            
            # r_i <= bounds_val
            # 1 * r_i <= bounds_val
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(bounds_val)
            
        # Pairwise constraints: r_i + r_j <= d_ij
        # 1*r_i + 1*r_j <= d_ij
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for r_i: >= 0
        bounds = [(0, None) for _ in range(n)]
        
        # Solve
        # linprog minimizes c^T x
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            radii = res.x
            return radii, -res.fun
        else:
            # Fallback: small radii
            radii = np.full(n, 1e-6)
            return radii, 0

    def objective(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        radii, total_r = solve_lp_radii(centers)
        # We want to maximize total_r, so minimize -total_r
        return -total_r

    # Initial guess
    init_centers = get_hexagonal_centers(n)
    x0 = init_centers.flatten()
    
    # Bounds for centers: [0, 1]
    bnds = [(0, 1) for _ in range(2 * n)]
    
    # Run optimization
    # SLSQP is good for constrained problems, but here we embedded constraints in LP.
    # So we just minimize objective.
    # However, objective is non-smooth.
    # Let's use differential_evolution for robustness.
    
    # To speed up, limit population size and maxiter
    # But we need accuracy.
    # Let's try a few iterations.
    
    # Since DE is stochastic, we might need a seed.
    res_de = differential_evolution(objective, bnds, seed=42, maxiter=100, popsize=15, tol=1e-7, polish=True)
    
    opt_centers = res_de.x.reshape(-1, 2)
    opt_radii, opt_sum = solve_lp_radii(opt_centers)
    
    # Validate and return
    # Ensure no NaN
    if np.isnan(opt_centers).any() or np.isnan(opt_radii).any():
        # Fallback to grid
        opt_centers = np.zeros((n, 2))
        opt_radii = np.zeros(n)
        idx = 0
        for r in range(5):
            for c in range(6): # 30 spots
                if idx < n:
                    opt_centers[idx] = [c * 0.2 + 0.1, r * 0.2 + 0.1]
                    opt_radii[idx] = 0.1
                    idx += 1
        # Correct radii for grid
        # Actually just return valid grid
        # 5x5 grid r=0.1
        # 26th circle?
        # Let's just return the optimized result if valid
        pass

    return opt_centers, opt_radii, float(opt_sum)

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
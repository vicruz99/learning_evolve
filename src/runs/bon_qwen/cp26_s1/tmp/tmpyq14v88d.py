import numpy as np
from scipy.optimize import linprog
import math

def get_hexagonal_grid(n, square_size=1.0, padding=0.05):
    """
    Generates a hexagonal grid of n points inside a square [0, square_size]^2.
    """
    points = []
    # Estimate row spacing and column spacing
    # Hexagonal packing density ~ 0.9069
    # Area per circle ~ pi * r^2. Total area ~ n * pi * r^2 <= square_size^2 * 0.9
    # r approx sqrt(0.9 * 1 / (n * pi))
    # Diameter approx 2r.
    # Let's just try to fit points.
    
    # Heuristic for spacing
    # Number of rows approx sqrt(n / (sqrt(3)/2)) * (square_size / diameter) ?
    # Let's just fill a grid.
    
    # Try to fit points in rows
    # Vertical spacing dy = sqrt(3)/2 * dx
    # Let's iterate on number of rows
    
    best_points = None
    
    # Try different row counts
    for num_rows in range(int(math.sqrt(n)), int(math.sqrt(n)) + 10):
        if num_rows == 0: continue
        dy = square_size / (num_rows + 1) # Approx
        # In hex, dy = dx * sqrt(3)/2 => dx = dy * 2/sqrt(3)
        dx = dy * 2 / math.sqrt(3)
        
        if dx <= 0: continue
        
        cols = int(square_size / dx) + 2
        
        current_points = []
        for r in range(num_rows):
            y = padding + (r + 1) * (square_size - 2*padding) / (num_rows + 1)
            # Shift every other row
            shift = 0 if r % 2 == 0 else dx / 2
            
            for c in range(cols):
                x = padding + shift + c * dx
                if x < 0 or x > square_size:
                    continue
                if len(current_points) >= n:
                    break
                current_points.append((x, y))
            if len(current_points) >= n:
                break
        if len(current_points) >= n:
            # Trim to n
            current_points = current_points[:n]
            # Check if this is better (e.g. more spread out?)
            # For now just take first valid
            best_points = current_points
            break
            
    if best_points is None:
        # Fallback to random
        np.random.seed(42)
        points = np.random.uniform(0.1, 0.9, (n, 2))
        return points

    points = np.array(best_points)
    # Clip to [0,1]
    points = np.clip(points, 0, 1)
    return points

def compute_approx_radius_sum(centers):
    """
    Computes a lower bound for the sum of radii assuming equal radii logic locally.
    Actually, just returns the max possible equal radius r such that 2r <= dist(i,j) and r <= dist(i, wall).
    Then sum = n * r.
    """
    n = centers.shape[0]
    min_dist = float('inf')
    
    # Distance to walls
    for i in range(n):
        x, y = centers[i]
        d_wall = min(x, 1-x, y, 1-y)
        if d_wall < min_dist:
            min_dist = d_wall
            
    # Distance between circles
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist * 2: # Optimization: if d is very small, it limits r
                 # Constraint is r_i + r_j <= d. If r_i=r_j=r, 2r <= d => r <= d/2.
                 pass
            # We track min(2r)
            half_dist = d / 2
            if half_dist < min_dist:
                min_dist = half_dist
                
    return n * min_dist

def solve_radii_lp(centers):
    """
    Given fixed centers, solve LP to maximize sum of radii.
    Variables: r_0, ..., r_{n-1}
    Maximize sum(r_i)
    Constraints:
      r_i <= x_i
      r_i <= 1 - x_i
      r_i <= y_i
      r_i <= 1 - y_i
      r_i + r_j <= dist(i, j)
      r_i >= 0
    """
    n = centers.shape[0]
    
    # Objective: max sum(r) => min -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ vars <= b_ub
    # r_i <= x_i  => 1*r_i <= x_i
    # r_i <= 1-x_i => 1*r_i <= 1-x_i
    # etc.
    # r_i + r_j <= dist => 1*r_i + 1*r_j <= dist
    
    A_ub = []
    b_ub = []
    
    # Wall constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(x)
        
        # r_i <= 1 - x
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - x)
        
        # r_i <= y
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(y)
        
        # r_i <= 1 - y
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - y)
        
    # Inter-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
            return radii, -res.fun
        else:
            # Fallback to equal radii if LP fails (unlikely)
            min_r = float('inf')
            for i in range(n):
                x, y = centers[i]
                d_wall = min(x, 1-x, y, 1-y)
                if d_wall < min_r: min_r = d_wall
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d/2 < min_r: min_r = d/2
            radii = np.full(n, min_r)
            return radii, n * min_r
    except Exception:
        # Fallback
        min_r = float('inf')
        for i in range(n):
            x, y = centers[i]
            d_wall = min(x, 1-x, y, 1-y)
            if d_wall < min_r: min_r = d_wall
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d/2 < min_r: min_r = d/2
        radii = np.full(n, min_r)
        return radii, n * min_r

def optimize_centers_force(centers, iterations=500):
    """
    Simple force-directed optimization to maximize minimum distance (approx).
    Uses repulsive forces.
    """
    n = centers.shape[0]
    centers = centers.copy()
    
    # Initial temperature / step size
    step = 0.01
    
    for t in range(iterations):
        forces = np.zeros_like(centers)
        
        # Compute distances
        # Force from walls
        for i in range(n):
            x, y = centers[i]
            # Wall forces: push away from 0 and 1
            # Potential ~ -log(dist)
            # Force ~ 1/dist^2 ?
            # Let's use linear repulsion for stability near boundary
            # If x < 0.1, push right. If x > 0.9, push left.
            if x < 0.1:
                forces[i, 0] += (0.1 - x) * 10
            elif x > 0.9:
                forces[i, 0] -= (x - 0.9) * 10
            
            if y < 0.1:
                forces[i, 1] += (0.1 - y) * 10
            elif y > 0.9:
                forces[i, 1] -= (y - 0.9) * 10
                
        # Inter-circle repulsion
        # Use a repulsive force that scales with overlap or inverse distance
        # We want to push apart if they are close.
        # Target distance? We don't know it.
        # But we can use a potential like 1/d^2.
        
        # To make it efficient, we can just sum vectors.
        # However, for 26 circles, O(N^2) is fine.
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    dist = 1e-6
                    diff = np.random.uniform(-0.01, 0.01, 2)
                
                # Repulsive force magnitude
                # F = k / dist^2
                # We want to push them apart to maximize spacing.
                # But we must not push them out of bounds too hard.
                # Let's use a force that is strong when close.
                # Maybe F = 1 / dist
                f_mag = 0.05 / (dist * dist) 
                
                # Direction
                dir_vec = diff / dist
                
                forces[i] += f_mag * dir_vec
                forces[j] -= f_mag * dir_vec
        
        # Apply forces
        # Normalize forces to step size?
        # Or just dampen.
        # To prevent explosion, clip forces.
        max_f = np.linalg.norm(forces, axis=1).max()
        if max_f > 0:
            # Scale forces to not move too much
            # Move amount proportional to force, but bounded
            move = forces * (step / (1 + np.linalg.norm(forces, axis=1, keepdims=True)))
        else:
            move = forces
            
        centers += move
        centers = np.clip(centers, 1e-6, 1 - 1e-6)
        
        # Decay step
        step *= 0.995

    return centers

def optimize_centers_nelder_mead(centers):
    """
    Use Nelder-Mead to maximize the minimum distance (approx).
    This helps in fine-tuning.
    """
    n = centers.shape[0]
    flat_centers = centers.flatten()
    
    def objective(flat_c):
        c = flat_c.reshape(-1, 2)
        # We want to maximize min(min_dist_wall, min_dist_pair / 2)
        # Equivalent to minimizing negative of that.
        
        min_d = float('inf')
        
        # Wall distances
        for i in range(n):
            x, y = c[i]
            d = min(x, 1-x, y, 1-y)
            if d < min_d:
                min_d = d
        
        # Pair distances / 2
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c[i] - c[j]) / 2
                if d < min_d:
                    min_d = d
        
        return -min_d # Minimize negative max radius
    
    # Bounds for Nelder-Mead? No, it doesn't support bounds directly.
    # We must ensure points stay in [0,1].
    # But Nelder-Mead can go outside.
    # We can use penalty in objective.
    
    def objective_with_penalty(flat_c):
        c = flat_c.reshape(-1, 2)
        # Penalty for being outside
        pen = 0
        for i in range(n):
            x, y = c[i]
            if x < 0 or x > 1 or y < 0 or y > 1:
                pen += 100 * (np.max([0, -x, x-1, -y, y-1]))
                # Actually just return huge cost
                return 1000 
            # Soft boundary penalty
            dist_wall = min(x, 1-x, y, 1-y)
            if dist_wall < 0.01:
                 # Penalty to keep away from walls slightly? 
                 # No, we want to allow touching walls.
                 pass
        
        val = objective(flat_c)
        return val

    # Since Nelder-Mead doesn't handle bounds, we rely on the fact that
    # going outside 0,1 is bad for distance to wall.
    # But it might explore outside.
    # Let's restrict initial simplex to valid region.
    
    # Actually, let's just use the force method result, it's usually good enough.
    # Or run a few iterations of random walk?
    return centers

def run_packing():
    n = 26
    
    # 1. Generate initial configurations
    # Try multiple seeds for hex grid
    configs = []
    
    # Config 1: Hexagonal grid
    np.random.seed(42)
    c1 = get_hexagonal_grid(n)
    configs.append(c1)
    
    # Config 2: Random
    np.random.seed(123)
    c2 = np.random.uniform(0.1, 0.9, (n, 2))
    configs.append(c2)
    
    # Config 3: Grid 5x5 + 1 random
    c3 = np.zeros((n, 2))
    idx = 0
    for x in np.linspace(0.1, 0.9, 5):
        for y in np.linspace(0.1, 0.9, 5):
            c3[idx] = [x, y]
            idx += 1
    # 26th circle
    # Try to place in a gap. Center (0.5, 0.5) is taken.
    # Gaps are at (0.2, 0.2), etc.
    # Let's place at (0.5, 0.5) but it's occupied.
    # Perturb the grid slightly?
    # Let's just put it at (0.5, 0.5) and let optimizer fix.
    # But (0.5, 0.5) is in grid.
    # Let's remove one and add one?
    # Or just random.
    c3[-1] = [0.5, 0.5] # Overlap, but optimizer will fix
    configs.append(c3)
    
    best_sum = -1
    best_centers = None
    best_radii = None
    
    for idx, centers in enumerate(configs):
        # 2. Optimize centers using force simulation
        opt_centers = optimize_centers_force(centers, iterations=1000)
        
        # 3. Solve LP for radii
        radii, current_sum = solve_radii_lp(opt_centers)
        
        # Check validity just in case (LP guarantees constraints, but numerical precision?)
        # validate_packing is read-only, we assume LP works.
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = opt_centers.copy()
            best_radii = radii.copy()
            
            # Try to improve further with local search on centers?
            # Since LP is sensitive to centers, maybe perturbation helps.
            # But LP finds optimal radii for fixed centers.
            # The function f(C) = max_sum_radii(C) is likely concave-ish?
            # We can try to perturb centers slightly around the good solution.
            
            # Let's try a few random perturbations
            for _ in range(10):
                perturbed = best_centers + np.random.normal(0, 0.01, best_centers.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                r_p, s_p = solve_radii_lp(perturbed)
                if s_p > best_sum:
                    best_sum = s_p
                    best_centers = perturbed
                    best_radii = r_p

    # Final validation check (mental)
    # Centers must be in [0,1], radii >= 0.
    # LP ensures r_i <= x_i etc. so centers+radii <= 1 is not guaranteed?
    # Wait. LP constraints: r_i <= x_i and r_i <= 1-x_i.
    # This implies x_i - r_i >= 0 and x_i + r_i <= 1.
    # Same for y.
    # And r_i + r_j <= dist.
    # So it is valid.
    
    return best_centers, best_radii, best_sum

# Helper functions must be top level.
# I defined them above.
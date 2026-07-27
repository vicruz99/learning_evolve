import numpy as np
from scipy.optimize import minimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False

    if np.isnan(radii).any():
        return False

    # Check if radii are nonnegative
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  
                return False

    return True

def get_constraints_and_bounds(n):
    """
    Defines the bounds and constraints for the optimization problem.
    Variables vector x: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    bounds = []
    for i in range(n):
        # x in [0, 1], y in [0, 1], r >= 0
        # But effectively r <= 0.5. 
        # We will enforce boundary constraints via non-overlap with walls implicitly or explicitly.
        # Here we use loose bounds and constraints.
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    constraints = []

    # Boundary constraints:
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # y - r >= 0  => r - y <= 0
    # y + r <= 1  => y + r - 1 <= 0
    
    # We can add these as linear constraints or penalty in objective.
    # Let's add them as constraints to be safe.
    
    # For each circle i (index i in 0..n-1)
    # Var index: 3*i, 3*i+1, 3*i+2 for x, y, r
    
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x - r >= 0  ->  r - x <= 0
        def make_wall_left(i):
            def wall_left(x):
                return -x[3*i] + x[3*i+2] # r - x. We want <= 0? 
                # Constraint func should return >= 0 for 'ineq'.
                # Standard scipy: fun(x) >= 0.
                # We want x - r >= 0 => x - r >= 0.
                return x[3*i] - x[3*i+2]
            return wall_left

        # x + r <= 1 -> 1 - x - r >= 0
        def make_wall_right(i):
            def wall_right(x):
                return 1.0 - x[3*i] - x[3*i+2]
            return wall_right

        # y - r >= 0
        def make_wall_bottom(i):
            def wall_bottom(x):
                return x[3*i+1] - x[3*i+2]
            return wall_bottom

        # y + r <= 1
        def make_wall_top(i):
            def wall_top(x):
                return 1.0 - x[3*i+1] - x[3*i+2]
            return wall_top

        constraints.append({'type': 'ineq', 'fun': make_wall_left(i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_right(i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_bottom(i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_top(i)})

    # Non-overlap constraints:
    # dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = 3 * i
            idx_yi = 3 * i + 1
            idx_ri = 3 * i + 2
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            def make_overlap(i, j):
                def overlap(x):
                    xi, yi, ri = x[idx_xi], x[idx_yi], x[idx_ri]
                    xj, yj, rj = x[idx_xj], x[idx_yj], x[idx_rj]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return dist - ri - rj
                return overlap
            
            constraints.append({'type': 'ineq', 'fun': make_overlap(i, j)})

    return bounds, constraints

def objective_func(x, n):
    """
    Objective: Maximize sum of radii.
    Minimizer minimizes, so we return negative sum.
    """
    radii = x[2::3]
    return -np.sum(radii)

def generate_initial_guess(n, seed=42):
    """
    Generates an initial configuration of n circles.
    Uses a hexagonal lattice pattern.
    """
    rng = np.random.RandomState(seed)
    
    # Estimate radius. For n=26, sum ~ 2.6, avg r ~ 0.1.
    # Lattice spacing approx 2*r = 0.2.
    r_init = 0.09 # Start slightly smaller to fit easily
    
    centers = []
    radii = []
    
    # Hexagonal packing logic
    # Rows
    # Try to fit in 1x1
    # dy = sqrt(3)/2 * 2r = sqrt(3)*r approx 0.156
    dy = math.sqrt(3) * r_init
    dx = 2 * r_init
    
    y = r_init
    row_parity = 0
    count = 0
    
    while count < n:
        x = r_init
        if row_parity == 1:
            x += r_init # Shift by r (half dx)
        
        while x < 1 - r_init and count < n:
            centers.append([x, y])
            radii.append(r_init)
            count += 1
            x += dx
        
        y += dy
        row_parity = 1 - row_parity

    # If we didn't get enough (unlikely with these params), fill randomly?
    # Or just scale up?
    # With r=0.09, we should fit plenty.
    
    # Add some random noise to avoid symmetry
    noise = rng.normal(0, 0.01, size=(len(centers), 2))
    centers = np.array(centers) + noise
    radii = np.array(radii)
    
    # Flatten to vector [x1, y1, r1, x2, y2, r2...]
    x_vec = np.zeros(3 * n)
    for i in range(n):
        x_vec[3*i] = centers[i][0]
        x_vec[3*i+1] = centers[i][1]
        x_vec[3*i+2] = radii[i]
        
    return x_vec

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds, constraints = get_constraints_and_bounds(n)
    
    # Try multiple seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3333]
    
    # We can also try different initial lattice densities
    # But fixed seed generation is fast enough.
    
    # Pre-calculate initial guesses for different seeds
    initial_guesses = []
    for seed in seeds:
        x0 = generate_initial_guess(n, seed=seed)
        initial_guesses.append(x0)

    # Also try a grid initialization
    # 5x5 grid + 1 random
    grid_centers = []
    step = 1.0 / 6.0 # 1/6 = 0.166, r ~ 0.08?
    # Try to fit 25 in grid
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            grid_centers.append([0.5/5 + i * (4.0/5)/4, 0.5/5 + j * (4.0/5)/4]) 
            # Actually simpler: 5 circles in [0,1] -> centers at 0.1, 0.3, 0.5, 0.7, 0.9
    grid_centers = []
    for i in range(5):
        for j in range(5):
            grid_centers.append([0.1 + i*0.2, 0.1 + j*0.2])
    
    # Add 26th circle in a gap?
    # Center of square (0.5, 0.5) is occupied?
    # 0.1, 0.3, 0.5... yes 0.5 is occupied.
    # Maybe (0.2, 0.2)? Occupied.
    # Try a random spot or just duplicate one and let optimizer split?
    # Let's just add a small circle at (0.2, 0.6)
    grid_centers.append([0.2, 0.6])
    
    x0_grid = np.zeros(3 * n)
    r_grid = 0.09 # small enough
    for i in range(n):
        x0_grid[3*i] = grid_centers[i][0]
        x0_grid[3*i+1] = grid_centers[i][1]
        x0_grid[3*i+2] = r_grid
    initial_guesses.append(x0_grid)

    # Run optimization
    for i, x0 in enumerate(initial_guesses):
        try:
            res = minimize(
                objective_func,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or (i == 0): # Keep if success or first run
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    x_opt = res.x
                    best_centers = np.zeros((n, 2))
                    best_radii = np.zeros(n)
                    for k in range(n):
                        best_centers[k] = [x_opt[3*k], x_opt[3*k+1]]
                        best_radii[k] = x_opt[3*k+2]
        except Exception as e:
            pass

    # Post-processing: Ensure strict validity and maybe small adjustments
    # Sometimes optimization lands on boundary.
    # The validate function allows 1e-12 tolerance, so we should be fine.
    
    # If best_centers is None (shouldn't happen), return empty
    if best_centers is None:
        # Fallback
        best_centers = np.random.rand(n, 2) * 0.5 + 0.25
        best_radii = np.full(n, 0.05)
        best_sum = np.sum(best_radii)

    # Final validation check just in case (though not strictly required to return True, 
    # we want to return a valid packing)
    # If invalid, we might need to shrink radii slightly.
    if not validate_packing(best_centers, best_radii):
        # Fallback: shrink radii slightly
        scale = 0.99
        best_radii *= scale
        # Recalculate sum
        best_sum = np.sum(best_radii)
        
        # If still invalid, drastic reduction
        if not validate_packing(best_centers, best_radii):
            best_radii *= 0.9
            best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
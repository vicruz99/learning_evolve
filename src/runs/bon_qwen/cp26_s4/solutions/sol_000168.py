# sol_000168 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 403fd447) state=1fb0f61c sum of radii=1.595735 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Objective function for maximizing minimum radius (equal radii assumption)
    # Returns negative minimum radius to be minimized
    def objective_equal_r(x_flat):
        c = x_flat.reshape(-1, 2)
        # Compute min distance to boundary
        # Distance to left/right walls: min(x, 1-x)
        # Distance to top/bottom walls: min(y, 1-y)
        dist_bound = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), 
                                np.minimum(c[:, 1], 1 - c[:, 1]))
        min_r = np.min(dist_bound)
        if min_r <= 0: return 0
        
        # Check inter-circle distances
        # We need min(dist(c_i, c_j) / 2) for all i != j
        # Optimized loop: compare circle i with i+1...n-1
        for i in range(n - 1):
            # Distances from circle i to circles i+1...n-1
            diffs = c[i+1:] - c[i]
            dists = np.linalg.norm(diffs, axis=1)
            # The minimum distance between i and any j>i
            min_d = np.min(dists)
            if min_d / 2 < min_r:
                min_r = min_d / 2
                if min_r <= 0: return 0
            
        return -min_r

    # Bounds for centers: [0, 1] for each coordinate
    bounds = [(0, 1)] * (2 * n)
    
    best_r = 0
    best_centers = None
    
    # Generate initial configurations to ensure robust search
    initializations = []
    
    # 1. Grid initialization (uniform distribution)
    grid = np.linspace(0.1, 0.9, 6) 
    pts = []
    for x in grid:
        for y in grid:
            pts.append([x, y])
    if len(pts) >= n:
        initializations.append(np.array(pts[:n]))
    
    # 2. Hexagonal-like initialization (dense packing heuristic)
    hex_centers = []
    r_guess = 0.09 # Estimated radius to fit 26 circles
    h = r_guess * np.sqrt(3) # Vertical spacing for hex packing
    idx = 0
    for row in range(10):
        # Alternating row lengths 5 and 4
        count = 5 if row % 2 == 0 else 4
        for col in range(count):
            if idx >= n: break
            # Shift odd rows horizontally by r_guess
            shift = r_guess if row % 2 != 0 else 0
            x = r_guess + shift + col * 2 * r_guess
            y = r_guess + row * h
            # Ensure within bounds [r, 1-r]
            if x <= 1 - r_guess and y <= 1 - r_guess:
                hex_centers.append([x, y])
                idx += 1
        if idx >= n: break
    if len(hex_centers) >= n:
        initializations.append(np.array(hex_centers[:n]))
        
    # 3. Random initialization (seed 1)
    np.random.seed(42)
    rand_centers = np.random.rand(n, 2) * 0.6 + 0.2
    initializations.append(rand_centers)
    
    # 4. Random initialization (seed 2)
    np.random.seed(1234)
    rand_centers2 = np.random.rand(n, 2) * 0.6 + 0.2
    initializations.append(rand_centers2)

    # Run optimization for each initialization to find best equal-radius packing
    for init_centers in initializations:
        x0 = init_centers.flatten()
        try:
            # Powell method is good for non-smooth objectives and bounded variables
            res = opt.minimize(objective_equal_r, x0, method='Powell', bounds=bounds, 
                               options={'maxiter': 2000, 'ftol': 1e-12})
            if -res.fun > best_r:
                best_r = -res.fun
                best_centers = res.x.reshape(-1, 2)
        except:
            pass

    # Fallback if optimization didn't yield a valid positive radius
    if best_centers is None or best_r <= 0:
        best_centers = np.random.rand(n, 2) * 0.8 + 0.1
        best_r = 0.05

    # Refine radii using Linear Programming for fixed centers
    # This allows for unequal radii to maximize the sum, potentially improving over equal radii.
    
    # Compute pairwise distances between centers
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Compute boundary limits for each center (distance to nearest wall)
    boundary_limits = np.array([
        min(best_centers[i, 0], 1 - best_centers[i, 0],
            best_centers[i, 1], 1 - best_centers[i, 1])
        for i in range(n)
    ])
    
    # LP Setup: Maximize sum(r_i) <=> Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints:
    # 1. r_i + r_j <= dists[i, j]  (Non-overlap condition)
    # 2. r_i <= boundary_limits[i]  (Boundary condition)
    # 3. r_i >= 0                   (Non-negative radius)
    
    n_constraints = n * (n - 1) // 2 + n
    A_ub = np.zeros((n_constraints, n))
    b_ub = np.zeros(n_constraints)
    
    k = 0
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[k, i] = 1
            A_ub[k, j] = 1
            b_ub[k] = dists[i, j]
            k += 1
            
    # Boundary constraints
    for i in range(n):
        A_ub[k, i] = 1
        b_ub[k] = boundary_limits[i]
        k += 1
        
    bounds_r = [(0, None)] * n
    
    from scipy.optimize import linprog
    try:
        # 'highs' is a robust linear programming solver available in recent SciPy versions
        res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    except ValueError:
        # Fallback to default method if 'highs' is not available
        res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r)
    
    if res_lp.success:
        radii = res_lp.x
    else:
        # Fallback to equal radii derived from the center optimization
        radii = np.full(n, best_r)
        
    # Safety clamp to ensure non-negative radii
    radii = np.maximum(radii, 0)
    
    # Final sum of radii
    sum_radii = np.sum(radii)
    
    return best_centers, radii, sum_radii

# sol_000246 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 01430d11) state=55a00ec0 sum of radii=2.215344 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_radii_lp(centers, n):
    """
    Solves the Linear Programming problem to maximize sum of radii
    given fixed centers.
    
    Constraints:
    1. r_i + r_j <= distance(c_i, c_j) for all i < j
    2. r_i <= x_i, r_i <= 1 - x_i, r_i <= y_i, r_i <= 1 - y_i
    3. r_i >= 0
    """
    # Precompute distances between all pairs
    # Centers shape (n, 2)
    # Diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Number of variables: n radii
    n_vars = n
    c_obj = -np.ones(n_vars) # Maximize sum => Minimize -sum
    
    # Inequality constraints A_ub @ x <= b_ub
    # 1. Pairwise distances: r_i + r_j <= dist_ij
    # We only need i < j to avoid redundancy, but linprog handles full matrix fine if sparse?
    # Actually, let's construct the matrix explicitly.
    # Number of pairwise constraints: n*(n-1)/2
    
    # Boundary constraints: 4 per circle
    # r_i <= x_i  => 1*r_i <= x_i
    # r_i <= 1-x_i => 1*r_i <= 1-x_i
    # r_i <= y_i  => 1*r_i <= y_i
    # r_i <= 1-y_i => 1*r_i <= 1-y_i
    
    n_pairwise = n * (n - 1) // 2
    n_boundary = 4 * n
    n_constraints = n_pairwise + n_boundary
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[row_idx, i] = 1.0
            A_ub[row_idx, j] = 1.0
            b_ub[row_idx] = dists[i, j]
            row_idx += 1
            
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = x
        row_idx += 1
        # r_i <= 1-x
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = 1.0 - x
        row_idx += 1
        # r_i <= y
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = y
        row_idx += 1
        # r_i <= 1-y
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = 1.0 - y
        row_idx += 1
        
    # Bounds for r_i: [0, infinity)
    bounds = [(0, None) for _ in range(n_vars)]
    
    # Solve LP
    # Using 'highs' method if available, else 'simplex'
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')
        except:
            # Fallback if scipy version is old
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if res.success:
        return -res.fun, res.x # Return sum and radii
    else:
        return 0.0, np.zeros(n_vars)

def generate_hexagonal_init(n):
    """
    Generates an initial hexagonal packing layout.
    """
    # Approximate radius for hexagonal packing in square
    # Area ~ 1, N circles. Area per circle ~ 1/N.
    # pi r^2 ~ 1/N * density_factor. 
    # Let's just place them in a grid and scale.
    
    # Rows and cols estimation
    # Hexagonal packing density is high.
    # Let's try to fit n circles.
    # Rows of alternating length.
    rows = []
    current_n = 0
    row_len = 6 # Start with a row length guess
    r_guess = 0.1
    
    # Heuristic to build rows
    # Try to fit roughly square aspect ratio
    cols_est = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    
    centers = []
    
    # Simple hex grid generation
    # Row height = sqrt(3)/2 * diameter
    # Col width = diameter
    
    # Let's just create a dense grid and pick first n
    # Or construct specifically.
    
    # Let's try a specific pattern for 26
    # 6, 5, 6, 5, 4 = 26
    pattern = [6, 5, 6, 5, 4]
    
    # Calculate dimensions required for this pattern with unit diameter 1
    # Width for 6 circles: 5 gaps * 1 + 2 radii = 6? 
    # Centers at 0.5, 1.5, ... 5.5. Width 6.
    # Actually distance between centers is 1.
    # Span from center of first to center of last is 5.
    # Plus radius on each side (0.5) -> Total width 6.
    
    # Height: 5 rows.
    # Vertical spacing sqrt(3)/2.
    # 4 gaps. Height = 4 * sqrt(3)/2 + 2*radius = 2*sqrt(3) + 1 ~ 4.46
    
    # Let's generate coordinates for this pattern with spacing 1 and vertical shift 1
    # Then normalize.
    
    y_coord = 0
    for i, count in enumerate(pattern):
        x_start = 0
        if i % 2 == 1: # Staggered rows shifted by 0.5
            x_start = 0.5
        for j in range(count):
            x = x_start + j
            centers.append([x, y_coord])
        y_coord += 1 # Vertical spacing in this raw coord
    
    centers = np.array(centers)
    
    # Normalize to [0,1]x[0,1]
    # Find min/max
    min_x, min_y = centers.min(axis=0)
    max_x, max_y = centers.max(axis=0)
    
    # Add padding to keep away from edges initially
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit inside [0.05, 0.95] roughly
    scale_x = 0.9 / width if width > 0 else 1
    scale_y = 0.9 / height if height > 0 else 1
    scale = min(scale_x, scale_y)
    
    centers = (centers - np.array([min_x, min_y])) * scale
    
    # Center in square
    cur_min = centers.min(axis=0)
    cur_max = centers.max(axis=0)
    shift_x = (1 - (cur_max[0] - cur_min[0])) / 2 - cur_min[0]
    shift_y = (1 - (cur_max[1] - cur_min[1])) / 2 - cur_min[1]
    
    centers += np.array([shift_x, shift_y])
    
    # If we didn't generate exactly 26, truncate or pad
    if len(centers) > n:
        centers = centers[:n]
    elif len(centers) < n:
        # Add random points
        extra = n - len(centers)
        for _ in range(extra):
            centers = np.vstack([centers, [np.random.rand(), np.random.rand()]])
            
    return centers

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialization
    centers = generate_hexagonal_init(n)
    
    # Initial solve to get a baseline
    best_sum, best_radii = solve_radii_lp(centers, n)
    best_centers = centers.copy()
    
    # 2. Local Search (Simulated Annealing)
    # We perturb centers and re-solve LP
    
    current_centers = centers.copy()
    current_sum = best_sum
    current_radii = best_radii
    
    temperature = 0.05 # Initial perturbation scale
    min_temp = 1e-6
    cooling_rate = 0.995
    
    # Number of iterations
    # We can run this in a loop. 
    # To be safe with time, let's do a fixed number of steps.
    # 5000 steps should be enough for 26 variables.
    n_iter = 2000
    
    for i in range(n_iter):
        # Perturb centers
        # Randomly select a subset of circles to move
        num_to_move = np.random.randint(1, 5)
        indices = np.random.choice(n, num_to_move, replace=False)
        
        new_centers = current_centers.copy()
        
        # Perturbation magnitude based on temperature
        # Move each selected center by a small random vector
        for idx in indices:
            delta = np.random.normal(0, temperature, 2)
            new_centers[idx] += delta
            
            # Project back to [0, 1]
            new_centers[idx] = np.clip(new_centers[idx], 0.0, 1.0)
            
        # Solve LP for new centers
        new_sum, new_radii = solve_radii_lp(new_centers, n)
        
        # Acceptance criteria
        # We always accept if better.
        # If worse, accept with probability exp((new - old)/temp)? 
        # But here objective is sum of radii.
        # Since LP gives exact max for fixed centers, we are climbing a rugged landscape.
        # Standard Simulated Annealing might help escape local optima.
        
        if new_sum > current_sum:
            current_sum = new_sum
            current_centers = new_centers
            current_radii = new_radii
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = current_centers.copy()
                best_radii = current_radii.copy()
        else:
            # Probabilistic acceptance
            delta_obj = new_sum - current_sum
            if delta_obj > 0 or np.random.rand() < np.exp(delta_obj / max(temperature, 1e-9)):
                current_sum = new_sum
                current_centers = new_centers
                current_radii = new_radii
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp

    # 3. Post-processing / Fine-tuning
    # Maybe try random restarts if stuck? 
    # Given the constraints, one run might be enough if init is good.
    # But let's try a few random restarts to be safe.
    
    # Actually, let's just run the optimization loop again with different seed
    # or perturb best_centers and run a short local search.
    
    # Let's do a few quick random restarts
    for restart in range(3):
        centers_restart = best_centers + np.random.normal(0, 0.01, (n, 2))
        centers_restart = np.clip(centers_restart, 0, 1)
        
        curr_c = centers_restart
        curr_s, curr_r = solve_radii_lp(curr_c, n)
        
        temp_r = 0.02
        for step in range(500):
            idx = np.random.randint(n)
            new_c = curr_c.copy()
            new_c[idx] += np.random.normal(0, temp_r, 2)
            new_c[idx] = np.clip(new_c[idx], 0, 1)
            
            ns, nr = solve_radii_lp(new_c, n)
            if ns > curr_s:
                curr_s, curr_r = ns, nr
                curr_c = new_c
                if ns > best_sum:
                    best_sum = ns
                    best_centers = new_c.copy()
                    best_radii = nr.copy()
            else:
                if np.random.rand() < np.exp((ns - curr_s)/max(temp_r, 1e-9)):
                    curr_s, curr_r = ns, nr
                    curr_c = new_c
            
            temp_r *= 0.995

    return best_centers, best_radii, best_sum

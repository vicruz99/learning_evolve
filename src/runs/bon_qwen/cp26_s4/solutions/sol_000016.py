# sol_000016 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 04e92922) state=10ab0b77 sum of radii=2.520242 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def generate_hexagonal_initialization(n):
    """Generate an initial hexagonal packing of n circles in a unit square."""
    centers = []
    r = 0.05 # Initial small radius
    row = 0
    col = 0
    
    # Hexagonal packing spacing
    dx = 1.5
    dy = 1.3
    
    # Fill the square with a hexagonal pattern
    y = 0.5
    while y <= 1.0:
        x = 0.5
        row_offset = 0.5 if row % 2 == 1 else 0
        while x <= 1.0 and len(centers) < n:
            centers.append([x + row_offset * 0.25, y])
            x += dx * 0.2
            col += 1
        y += dy * 0.2
        row += 1
        col = 0

    # Pad if we didn't get n circles, or trim if we did
    centers = centers[:n]
    while len(centers) < n:
        centers.append([0.5, 0.5])
        
    return np.array(centers)

def objective_and_constraints(centers_radii, n):
    """
    Computes the objective (negative sum of radii) and constraint violations
    for the optimization solver.
    """
    centers = centers_radii[:2 * n].reshape(n, 2)
    radii = centers_radii[2 * n:]
    
    # Objective: Maximize sum of radii -> Minimize negative sum
    obj = -np.sum(radii)
    
    # Constraints list for SLSQP
    constraints = []
    
    # 1. Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # This is equivalent to: x - r >= 0, 1 - x - r >= 0, etc.
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        constraints.append(x - r)          # x >= r
        constraints.append(1 - x - r)      # x <= 1-r
        constraints.append(y - r)          # y >= r
        constraints.append(1 - y - r)      # y <= 1-r
        
    # 2. Non-overlap constraints: dist(i, j) >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2 -> dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            sum_r = radii[i] + radii[j]
            constraints.append(dist_sq - sum_r**2)
            
    return obj, constraints

def solve_lp_radii(centers):
    """
    Given fixed centers, solve for maximum radii using Linear Programming.
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints matrix A_ub * r <= b_ub
    # 1. Boundary constraints:
    # r_i <= x_i  => r_i - x_i <= 0? No.
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # And r_i >= 0
    
    # 2. Overlap constraints: r_i + r_j <= dist(i, j)
    
    A_ub = []
    b_ub = []
    
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
    
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback to small radii if LP fails
        return np.full(n, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)
    
    # 1. Initialize centers
    init_centers = generate_hexagonal_initialization(n)
    init_radii = np.full(n, 0.04)
    
    # Flatten for SLSQP
    x0 = np.concatenate([init_centers.flatten(), init_radii])
    
    # 2. Non-linear Optimization
    def obj_func(vars):
        obj, _ = objective_and_constraints(vars, n)
        return obj
    
    def con_func(vars):
        _, con = objective_and_constraints(vars, n)
        return con
    
    # Constraints must be >= 0 for SLSQP inequality constraints?
    # SLSQP convention: constr(x) >= 0 for 'ineq'
    # My con_func returns values that must be >= 0.
    constraints_slsqp = {'type': 'ineq', 'fun': con_func}
    
    # Bounds for centers [0, 1] and radii [0, 0.5]
    bounds_slsqp = []
    for _ in range(n):
        bounds_slsqp.append((0.0, 1.0)) # x
        bounds_slsqp.append((0.0, 1.0)) # y
    for _ in range(n):
        bounds_slsqp.append((0.0, 0.5)) # r
        
    options = {'maxiter': 1000, 'ftol': 1e-9}
    res_nlp = minimize(obj_func, x0, method='SLSQP', bounds=bounds_slsqp, 
                       constraints=constraints_slsqp, options=options)
    
    best_centers = res_nlp.x[:2*n].reshape(n, 2)
    
    # 3. Linear Programming Refinement
    # Fix centers and find max radii
    final_radii = solve_lp_radii(best_centers)
    
    # 4. Verification and slight adjustment if LP finds radii that violate 
    # the implicit non-overlap due to numerical precision in LP setup
    # LP handles r_i + r_j <= dist, so it should be safe.
    
    sum_radii = np.sum(final_radii)
    
    return best_centers, final_radii, sum_radii

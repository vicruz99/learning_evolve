# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f550adc) state=71bdef39 sum of radii=2.402606 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_radii_lp(centers):
    """
    Solves the Linear Programming problem to maximize sum of radii 
    given fixed centers.
    
    Maximize sum(r_i)
    Subject to:
      r_i + r_j <= dist(centers[i], centers[j]) for all i < j
      r_i <= boundary_limit(centers[i])
      r_i >= 0
    """
    n = centers.shape[0]
    
    # Objective: minimize -sum(r_i)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # 1. Pairwise distance constraints: r_i + r_j <= d_ij
    # 2. Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    
    num_constraints = (n * (n - 1)) // 2 + n
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    # Pairwise constraints
    constraint_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            A_ub[constraint_idx, i] = 1.0
            A_ub[constraint_idx, j] = 1.0
            b_ub[constraint_idx] = dist
            constraint_idx += 1
            
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
        # This is equivalent to r_i <= min(x, 1-x, y, 1-y)
        limit = min(x, 1 - x, y, 1 - y)
        A_ub[constraint_idx, i] = 1.0
        b_ub[constraint_idx] = limit
        constraint_idx += 1
        
    # Bounds for r_i: r_i >= 0. 
    # Since we want to maximize sum, and constraints are upper bounds, 
    # lower bound 0 is natural.
    bounds = [(0, None)] * n
    
    # Solve LP
    # Use 'highs' solver if available, it's robust. 
    # Fallback to 'interior-point' or 'simplex' if needed, but highs is standard in newer scipy.
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except ValueError:
        # Fallback for older scipy versions
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        
    if res.success:
        return -res.fun, res.x
    else:
        # Fallback: return small radii if LP fails (should not happen for valid centers)
        return 0.0, np.zeros(n)

def run_packing():
    np.random.seed(42) # For reproducibility
    n = 26
    
    # Initialize centers: Hexagonal-ish grid perturbed
    # A 5x5 grid has 25 points. We need 26.
    # Let's place 25 in a 5x5 grid and one in the center of a gap or random.
    # Actually, random initialization with some spacing is often better for SA.
    
    # Generate points on a grid then perturb
    # Grid 5x5 points at 0.1, 0.3, 0.5, 0.7, 0.9
    grid_points = []
    for i in range(5):
        for j in range(5):
            grid_points.append([0.1 + 0.2*i, 0.1 + 0.2*j])
    
    # We have 25 points. Add one more at (0.2, 0.6) - roughly center of a hole?
    # Or just random. Let's try to spread them.
    # A better initial config is just random in the square, but avoiding edges initially?
    # Let's use the grid + 1 random point.
    extra_point = [0.2 + 0.1*np.random.rand(), 0.6 + 0.1*np.random.rand()]
    # Actually, let's just generate 26 random points inside [0.1, 0.9] to be safe initially
    centers = np.random.uniform(0.1, 0.9, size=(n, 2))
    
    # Initial evaluation
    current_sum_radii, current_radii = solve_radii_lp(centers)
    
    # Simulated Annealing parameters
    T = 0.1 # Initial temperature
    alpha = 0.995 # Cooling rate
    iterations = 5000 # Number of iterations
    
    best_sum_radii = current_sum_radii
    best_centers = centers.copy()
    best_radii = current_radii.copy()
    
    for it in range(iterations):
        # Pick a random circle to move
        idx = np.random.randint(0, n)
        
        # Propose a move: Gaussian perturbation
        # Scale of move depends on temperature? Or fixed?
        # Let's use fixed scale initially, maybe decrease.
        step_size = 0.05 * (0.5 + 0.5 * T) 
        delta = np.random.normal(0, step_size, size=2)
        
        new_centers = centers.copy()
        new_centers[idx] += delta
        
        # Clamp to [0, 1] to ensure centers stay in square (radii will handle boundary)
        # Actually centers must be in [0,1]. 
        # If center goes outside, radii might be forced to 0 or negative? 
        # LP handles r_i <= min(x, 1-x...). If x < 0, limit is negative, r_i <= negative -> r_i=0 (since r_i>=0).
        # So clamping is good practice but not strictly necessary if LP is robust.
        new_centers = np.clip(new_centers, 1e-4, 1 - 1e-4)
        
        # Evaluate new configuration
        new_sum_radii, new_radii = solve_radii_lp(new_centers)
        
        # Acceptance criterion
        delta_E = new_sum_radii - current_sum_radii
        if delta_E > 0 or np.random.rand() < np.exp(delta_E / (T + 1e-9)):
            centers = new_centers
            current_sum_radii = new_sum_radii
            current_radii = new_radii
            
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_centers = centers.copy()
                best_radii = current_radii.copy()
        
        # Cool down
        T *= alpha
        
    # Final polish: Run LP one last time on best centers to ensure radii are optimal
    final_sum, final_radii = solve_radii_lp(best_centers)
    
    # The LP solution might have tiny numerical errors making dist < r1+r2 by epsilon
    # The validation function allows 1e-12 tolerance.
    # However, to be safe, we can slightly shrink radii if needed, but LP should be exact.
    # Let's just return.
    
    return best_centers, final_radii, final_sum

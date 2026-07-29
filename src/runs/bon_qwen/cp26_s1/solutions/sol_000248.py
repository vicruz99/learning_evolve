# sol_000248 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 01430d11) state=03c6cebd sum of radii=2.626582 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_optimal_radii_and_forces(centers):
    """
    Solves the LP for radii and returns optimal radii and dual variables (marginals).
    Centers: np.array of shape (n, 2)
    Returns: radii, pair_marginals, boundary_marginals
    """
    n = centers.shape[0]
    
    # Constraints: A_ub * r <= b_ub
    # Objective: min -sum(r)  => c = -1
    
    # 1. Pairwise constraints: r_i + r_j <= dist_ij
    # We will construct the A matrix and b vector.
    # Order of constraints:
    #   0 to N_pairs-1: Pairwise constraints
    #   N_pairs to N_pairs+N-1: Boundary constraints
    
    pairs = []
    A_rows = []
    b_vals = []
    
    pair_indices = [] # To map back marginals to pairs
    
    p_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            pair_indices.append(p_idx)
            p_idx += 1
            
    num_pairs = len(pairs)
    num_boundaries = n
    
    A_ub = np.zeros((num_pairs + num_boundaries, n))
    b_ub = np.zeros(num_pairs + num_boundaries)
    
    # Fill pairwise constraints
    for idx, (i, j) in enumerate(pairs):
        dist = np.linalg.norm(centers[i] - centers[j])
        A_ub[idx, i] = 1
        A_ub[idx, j] = 1
        b_ub[idx] = dist
        # Store indices in a way to retrieve later if needed, 
        # but we can just iterate pairs again or store mapping.
        # Let's just rely on the order being consistent.
        
    # Fill boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        row_idx = num_pairs + i
        x, y = centers[i]
        dist_boundary = min(x, 1 - x, y, 1 - y)
        A_ub[row_idx, i] = 1
        b_ub[row_idx] = dist_boundary
        
    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Using 'highs' solver is usually fast and robust
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback if LP fails (should not happen with valid centers)
        return np.zeros(n), {}, {}
        
    radii = res.x
    
    # Extract marginals (dual variables)
    # Marginals for inequality constraints correspond to A_ub rows.
    marginals = res.ineqlin.marginals
    
    pair_marginals = {}
    boundary_marginals = np.zeros(n)
    
    for idx, (i, j) in enumerate(pairs):
        # Marginal for pair constraint
        # In minimization, marginal <= 0 usually if constraint is active and restricting.
        # We take absolute value or negate for force magnitude.
        pair_marginals[(i, j)] = marginals[idx]
        
    for i in range(n):
        boundary_marginals[i] = marginals[num_pairs + i]
        
    return radii, pair_marginals, boundary_marginals

def run_packing():
    n = 26
    
    # 1. Initialize centers with a perturbed hexagonal grid
    centers = np.zeros((n, 2))
    
    # Try to fit roughly 5x5 or 6x4 grid pattern
    # Hexagonal packing: rows spaced by sqrt(3)/2 * width
    # Let's just use a dense grid and let optimization spread them out
    
    # Simple grid initialization
    # 5 rows, 6 cols = 30 points, take 26? 
    # Or 5 rows of 5 (25) + 1 center?
    # Let's do a 6x5 grid pattern
    
    rows = 6
    cols = 5 # 6*5 = 30, too many. 
    # Let's do rows with varying cols to get 26.
    # 5, 5, 5, 5, 6? Sum = 26.
    
    row_counts = [5, 5, 5, 5, 6] # Wait 5+5+5+5+6 = 26.
    # Actually 5 rows.
    # Let's shift odd rows.
    
    # Coordinates
    x_spacing = 1.0 / (max(row_counts) + 1) # approx
    y_spacing = 1.0 / (len(row_counts) + 1)
    
    idx = 0
    for r in range(len(row_counts)):
        count = row_counts[r]
        # Shift x for odd rows to make it hexagonal-like
        shift = 0.5 * x_spacing if r % 2 == 1 else 0
        
        # Distribute x in [x_spacing, 1-x_spacing]
        # Actually just uniform distribution
        # To avoid boundary issues, keep away from 0 and 1 initially
        start_x = 0.1
        end_x = 0.9
        span = end_x - start_x
        
        for c in range(count):
            x = start_x + (c + 0.5 + shift) * (span / (count + 0.5)) 
            # Simple linear spacing
            x = start_x + c * (span / (count - 1)) if count > 1 else 0.5
            
            # Clamp x
            x = np.clip(x, 0.05, 0.95)
            
            y = 0.1 + r * (0.8 / (len(row_counts) - 1)) if len(row_counts) > 1 else 0.5
            y = np.clip(y, 0.05, 0.95)
            
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            
    # Add small random perturbation to break symmetry
    centers += np.random.normal(0, 0.02, centers.shape)
    centers = np.clip(centers, 1e-5, 1 - 1e-5)
    
    # 2. Optimization Loop
    current_radii = np.zeros(n)
    best_sum_radii = 0
    best_centers = centers.copy()
    best_radii = current_radii.copy()
    
    # Parameters
    step_size = 0.05
    decay = 0.995
    num_iterations = 2000
    
    for iter in range(num_iterations):
        # Solve LP
        radii, pair_margs, bound_margs = get_optimal_radii_and_forces(centers)
        sum_r = np.sum(radii)
        
        if sum_r > best_sum_radii:
            best_sum_radii = sum_r
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pair forces
        # Marginal lambda for constraint r_i + r_j <= d_ij
        # If lambda < 0 (in min problem), it means relaxing constraint (increasing d_ij) reduces obj (increases sum r).
        # We want to increase d_ij.
        # Force on i is proportional to -lambda * (c_i - c_j) / d_ij
        
        # Since linprog minimizes -sum(r), and constraint is r_i+r_j <= d_ij.
        # Relaxing d_ij (increasing it) allows larger r, so sum(r) increases, -sum(r) decreases.
        # So obj decreases. Marginal should be <= 0.
        # We use weight = -marginal.
        
        for i in range(n):
            for j in range(i + 1, n):
                m = pair_margs[(i, j)]
                if m < -1e-9: # Active constraint
                    weight = -m
                    vec = centers[i] - centers[j]
                    dist = np.linalg.norm(vec)
                    if dist > 1e-9:
                        unit_vec = vec / dist
                        forces[i] += weight * unit_vec
                        forces[j] -= weight * unit_vec
        
        # Boundary forces
        # Constraint r_i <= b_i.
        # Relaxing b_i (increasing distance to boundary) helps.
        # Marginal should be <= 0.
        # Weight = -marginal.
        # Direction: gradient of b_i.
        # b_i = min(x, 1-x, y, 1-y).
        # If x is min, b_i=x, grad=(1,0). Move +x.
        # If 1-x is min, b_i=1-x, grad=(-1,0). Move -x.
        # Basically move away from the closest wall.
        
        for i in range(n):
            m = bound_margs[i]
            if m < -1e-9:
                weight = -m
                x, y = centers[i]
                
                # Find closest boundary
                dists = [x, 1-x, y, 1-y]
                min_idx = np.argmin(dists)
                
                # Direction to increase distance
                # 0: x (move +x)
                # 1: 1-x (move -x)
                # 2: y (move +y)
                # 3: 1-y (move -y)
                
                if min_idx == 0:
                    forces[i, 0] += weight
                elif min_idx == 1:
                    forces[i, 0] -= weight
                elif min_idx == 2:
                    forces[i, 1] += weight
                elif min_idx == 3:
                    forces[i, 1] -= weight
                    
        # Update centers
        # Apply forces
        centers += step_size * forces
        
        # Clip to valid range [0, 1]
        # Ideally keep small margin to avoid numerical issues with boundary dist
        centers = np.clip(centers, 1e-6, 1 - 1e-6)
        
        # Decay step size
        step_size *= decay
        
        # Random restart / perturbation occasionally to escape local minima
        if iter % 200 == 0 and iter > 0:
            centers += np.random.normal(0, 0.01, centers.shape)
            centers = np.clip(centers, 1e-6, 1 - 1e-6)

    return best_centers, best_radii, best_sum_radii

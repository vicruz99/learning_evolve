# sol_000188 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9afef83a) state=a62d3615 sum of radii=1.771592 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import warnings

# Suppress warnings from linprog for cleaner output
warnings.filterwarnings('ignore')

def compute_forces(centers, marginals, n_pairs_map):
    """
    Computes forces on centers based on LP dual variables (marginals).
    Forces are derived from the sensitivity of the objective (sum of radii) 
    to changes in the RHS of the constraints (which depend on centers).
    """
    n = centers.shape[0]
    forces_x = np.zeros(n)
    forces_y = np.zeros(n)
    
    # 1. Boundary constraints forces
    # Marginals indices:
    # 0..n-1: r_i <= x_i
    # n..2n-1: r_i <= 1 - x_i
    # 2n..3n-1: r_i <= y_i
    # 3n..4n-1: r_i <= 1 - y_i
    
    # r_i <= x_i (RHS is x_i, deriv wrt x_i is 1)
    for i in range(n):
        lam = marginals[i]
        forces_x[i] += lam
        
    # r_i <= 1 - x_i (RHS is 1-x_i, deriv wrt x_i is -1)
    for i in range(n):
        lam = marginals[n + i]
        forces_x[i] -= lam
        
    # r_i <= y_i (RHS is y_i, deriv wrt y_i is 1)
    for i in range(n):
        lam = marginals[2*n + i]
        forces_y[i] += lam
        
    # r_i <= 1 - y_i (RHS is 1-y_i, deriv wrt y_i is -1)
    for i in range(n):
        lam = marginals[3*n + i]
        forces_y[i] -= lam

    # 2. Pairwise constraints forces
    # r_i + r_j <= dist_ij
    # RHS is dist_ij. 
    # dist_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
    # grad(dist_ij) wrt xi = (xi - xj) / dist_ij
    # grad(dist_ij) wrt xj = (xj - xi) / dist_ij
    
    # Marginals for pairs start at index 4*n
    # We iterate through the pairs we stored
    
    base_idx = 4 * n
    for idx, (i, j) in enumerate(n_pairs_map):
        lam = marginals[base_idx + idx]
        if lam > 1e-9: # Only consider active constraints
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 1e-9:
                # Force proportional to lambda * gradient of distance
                # We want to maximize sum radii, so we move in direction of increasing dist
                # Force on i is + lam * grad_i
                # Force on j is + lam * grad_j
                
                fx = lam * dx / dist
                fy = lam * dy / dist
                
                forces_x[i] += fx
                forces_y[i] += fy
                
                forces_x[j] -= fx
                forces_y[j] -= fy
                
    return forces_x, forces_y

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers
    # Start with a grid and add perturbation to break symmetry
    # 5x5 grid is 25 points. We need 26.
    # Let's use a 6x5 grid logic or just random dense packing
    # A good heuristic is to pack them as densely as possible initially
    
    # Create a grid of 26 points
    # Maybe 6 rows: 4, 5, 4, 5, 4, 4 ?
    # Or just random
    np.random.seed(42)
    
    # Initialize with a perturbed grid
    # 5 columns, 6 rows roughly
    cols = 5
    rows = 6
    # Spacing
    dx = 1.0 / (cols + 1)
    dy = 1.0 / (rows + 1)
    
    centers = []
    for r in range(rows):
        for c in range(cols):
            centers.append([
                (c + 1) * dx + np.random.uniform(-0.05, 0.05),
                (r + 1) * dy + np.random.uniform(-0.05, 0.05)
            ])
    centers = np.array(centers)
    
    # If we have more than 26, slice (we generated 30)
    # Actually 5*6 = 30.
    centers = centers[:n]
    
    # Ensure centers are within reasonable bounds [0.1, 0.9]
    centers = np.clip(centers, 0.1, 0.9)

    # Precompute pair indices
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    num_pairs = len(pairs)
    num_constraints = 4 * n + num_pairs
    
    # Precompute A_ub structure for pairs (constant)
    # A_ub is sparse-ish but for n=26 dense is fine
    # Actually A_ub changes only in b_ub (RHS) for pairs, 
    # but the rows for pairs are constant (1 at i, 1 at j).
    
    # Build constant part of A_ub for pairs
    # Shape (num_pairs, n)
    A_pairs = np.zeros((num_pairs, n))
    for idx, (i, j) in enumerate(pairs):
        A_pairs[idx, i] = 1.0
        A_pairs[idx, j] = 1.0
        
    # Build A_ub for boundaries (identity matrices)
    # We can construct the full A_ub matrix
    # Rows 0..n-1: I (r_i <= x_i)
    # Rows n..2n-1: I (r_i <= 1-x_i)
    # Rows 2n..3n-1: I (r_i <= y_i)
    # Rows 3n..4n-1: I (r_i <= 1-y_i)
    # Rows 4n..end: A_pairs
    
    A_ub = np.vstack([
        np.eye(n),
        np.eye(n),
        np.eye(n),
        np.eye(n),
        A_pairs
    ])
    
    c = -np.ones(n) # Maximize sum -> minimize negative sum
    bounds = [(0, None) for _ in range(n)]
    
    # Optimization parameters
    max_iter = 500
    step_size = 0.01
    alpha = 0.9 # Momentum / damping
    
    best_sum_radii = -1.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    # Previous velocity for momentum
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    for iteration in range(max_iter):
        # 1. Solve LP for radii given current centers
        b_ub = np.zeros(num_constraints)
        
        # Fill boundary RHS
        for i in range(n):
            x, y = centers[i]
            b_ub[i] = x
            b_ub[n + i] = 1.0 - x
            b_ub[2*n + i] = y
            b_ub[3*n + i] = 1.0 - y
            
        # Fill pair RHS
        for idx, (i, j) in enumerate(pairs):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            b_ub[4*n + idx] = np.sqrt(dx*dx + dy*dy)
            
        # Solve
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                # If LP fails, try to recover or break
                # Sometimes happens if constraints are infeasible (shouldn't be for valid centers)
                # But if centers are too close? No, radii can be 0.
                # Maybe numerical issue.
                continue
                
            radii = res.x
            current_sum = np.sum(radii)
            
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
            # 2. Compute forces for next iteration
            # We need marginals. Highs provides them in res.marginals
            # Note: marginals order matches constraints order
            
            if hasattr(res, 'marginals'):
                m = res.marginals
                # Inequalities marginals
                # Note: scipy linprog marginals for inequalities correspond to A_ub x <= b_ub
                ineq_marginals = m.ineqlin
                
                # Check if marginals are available (sometimes None or 0 if not requested/supported)
                if ineq_marginals is not None:
                    fx, fy = compute_forces(centers, ineq_marginals, pairs)
                    
                    # Update velocities with momentum
                    vx = alpha * vx + (1 - alpha) * fx
                    vy = alpha * vy + (1 - alpha) * fy
                    
                    # Update centers
                    # Adaptive step size?
                    # Decrease step size over time to converge
                    current_step = step_size * (1.0 / (1.0 + iteration * 0.01))
                    
                    centers[:, 0] += current_step * vx
                    centers[:, 1] += current_step * vy
                    
                    # Clamp centers to stay inside [r_min, 1-r_min]
                    # Since min radius can be 0, strictly [0, 1] is safe, 
                    # but to allow non-zero radii, keep away from boundary slightly?
                    # Actually, if center hits 0, radius becomes 0.
                    # It's valid. But let's keep it in [0.001, 0.999] to avoid numerical issues with 0 dist
                    centers = np.clip(centers, 0.001, 0.999)
                    
        except Exception as e:
            # Fallback if optimization fails
            continue

    return best_centers, best_radii, best_sum_radii

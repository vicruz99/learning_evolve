import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import pdist, squareform

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal packing
    centers = np.zeros((n_circles, 2))
    radius_init = 0.09
    # Arrange in rows
    row = 0
    col = 0
    idx = 0
    x, y = 0.5, 0.5 # Center start
    # Just creating a compact cluster to start, or grid
    # Let's do a dense grid first, it's safer for convergence
    # 5x5 grid covers 25, add 1 in middle
    positions = []
    for r in range(5):
        for c in range(5):
            positions.append((0.1 + c * 0.2, 0.1 + r * 0.2))
    # 26th circle
    positions.append((0.5, 0.5))
    
    # Better initialization: Hexagonal lattice
    # Approximate radius 0.1
    # 5 rows, 5-6-5-6-4? 
    # Let's just use a grid perturbation
    centers = np.array(positions)
    
    # Optimization parameters
    n_iterations = 2000
    step_size = 0.01
    decay_rate = 0.995
    
    # Precompute pair indices for constraints
    pair_indices = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            pair_indices.append((i, j))
    n_pairs = len(pair_indices)
    
    # Constraint matrix for LP: 
    # Variables: r_0, r_1, ..., r_25
    # Constraints:
    # 1. r_i <= x_i
    # 2. r_i <= 1 - x_i
    # 3. r_i <= y_i
    # 4. r_i <= 1 - y_i
    # 5. r_i + r_j <= dist(i, j)
    
    # Total constraints: 4*n + n_pairs
    n_constraints = 4 * n_circles + n_pairs
    
    # Build A_ub matrix once, b_ub updates
    # But b_ub depends on centers, so we rebuild or update b_ub.
    # Rebuilding is fast for N=26.
    
    for _ in range(n_iterations):
        # Solve LP for radii
        # Objective: max sum(r) => min -sum(r)
        c_obj = -np.ones(n_circles)
        
        A_ub = np.zeros((n_constraints, n_circles))
        b_ub = np.zeros(n_constraints)
        
        # Boundary constraints
        for i in range(n_circles):
            row_idx = i # 0..25 for x_i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = centers[i, 0]
            
            row_idx = n_circles + i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = 1.0 - centers[i, 0]
            
            row_idx = 2 * n_circles + i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = centers[i, 1]
            
            row_idx = 3 * n_circles + i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = 1.0 - centers[i, 1]
            
        # Pairwise constraints
        dists = squareform(pdist(centers))
        for k, (i, j) in enumerate(pair_indices):
            row_idx = 4 * n_circles + k
            A_ub[row_idx, i] = 1.0
            A_ub[row_idx, j] = 1.0
            b_ub[row_idx] = dists[i, j]
            
        # Bounds for r_i >= 0
        bounds = [(0, None) for _ in range(n_circles)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                break
            radii = res.x
            
            # Get dual variables (marginals) for forces
            # In scipy linprog, duals are stored in res.ineqlin.marginals
            # Order matches constraints in A_ub
            
            marginals = res.ineqlin.marginals
            
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Boundary forces
            # Constraint 0..25: r_i <= x_i. Marginal mu.
            # d(sum)/d(x_i) = mu. Force pushes +x if mu > 0
            # Wait, if r_i <= x_i is active, increasing x_i allows larger r_i.
            # So we want to increase x_i.
            for i in range(n_circles):
                # x constraint
                mu = marginals[i]
                forces[i, 0] += mu
                
                # 1-x constraint: r_i <= 1 - x_i => r_i + x_i <= 1.
                # Constraint form in A_ub: 1*r_i <= 1 - x_i.
                # b_ub depends on x_i. b_ub = 1 - x_i.
                # Sensitivity wrt b_ub is mu.
                # d(sum)/d(x_i) = d(sum)/d(b_ub) * d(b_ub)/d(x_i) = mu * (-1).
                # So force is -mu.
                mu = marginals[n_circles + i]
                forces[i, 0] -= mu
                
                # y constraint
                mu = marginals[2 * n_circles + i]
                forces[i, 1] += mu
                
                # 1-y constraint
                mu = marginals[3 * n_circles + i]
                forces[i, 1] -= mu
            
            # Pairwise forces
            # Constraint: r_i + r_j <= d_{ij}. b_ub = d_{ij}.
            # Marginal mu. d(sum)/d(d_{ij}) = mu.
            # d_{ij} = ||c_i - c_j||.
            # Gradient of d_{ij} wrt c_i is (c_i - c_j) / d_{ij}.
            # Force on i: mu * (c_i - c_j) / d_{ij}.
            # Force on j: mu * (c_j - c_i) / d_{ij}.
            
            for k, (i, j) in enumerate(pair_indices):
                mu = marginals[4 * n_circles + k]
                if mu > 1e-9: # Only active constraints matter
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d > 1e-9:
                        fx = mu * dx / d
                        fy = mu * dy / d
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Apply forces to centers
            centers += step_size * forces
            
            # Project centers back into [0, 1] to prevent escape
            # Though forces should keep them in, numerical stability helps
            centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
            
            step_size *= decay_rate
            
        except Exception as e:
            break

    # Final radii calculation via LP one last time
    c_obj = -np.ones(n_circles)
    A_ub = np.zeros((n_constraints, n_circles))
    b_ub = np.zeros(n_constraints)
    
    for i in range(n_circles):
        A_ub[i, i] = 1.0
        b_ub[i] = centers[i, 0]
        A_ub[n_circles + i, i] = 1.0
        b_ub[n_circles + i] = 1.0 - centers[i, 0]
        A_ub[2 * n_circles + i, i] = 1.0
        b_ub[2 * n_circles + i] = centers[i, 1]
        A_ub[3 * n_circles + i, i] = 1.0
        b_ub[3 * n_circles + i] = 1.0 - centers[i, 1]
        
    dists = squareform(pdist(centers))
    for k, (i, j) in enumerate(pair_indices):
        row_idx = 4 * n_circles + k
        A_ub[row_idx, i] = 1.0
        A_ub[row_idx, j] = 1.0
        b_ub[row_idx] = dists[i, j]
        
    bounds = [(0, None) for _ in range(n_circles)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    final_radii = res.x
    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, float(sum_radii)
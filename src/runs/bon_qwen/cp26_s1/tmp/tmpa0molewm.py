import numpy as np
from scipy.optimize import minimize

def compute_objective(params, n, lam):
    """
    Computes the penalized objective for circle packing.
    Maximizes radius r while penalizing overlaps and boundary violations.
    """
    centers = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    # Calculate pairwise distances efficiently
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Ignore self-distances by setting diagonal to a large value
    np.fill_diagonal(dists, 2.0)
    
    # Overlap penalty: penalize when distance < 2r
    viol_overlap = np.maximum(0.0, 2.0*r - dists)
    pen_overlap = np.sum(viol_overlap**2)
    
    # Boundary penalties: penalize when circle crosses [0,1]
    pen_bound = np.sum(np.maximum(0.0, r - centers[:, 0])**2)
    pen_bound += np.sum(np.maximum(0.0, r - (1.0 - centers[:, 0]))**2)
    pen_bound += np.sum(np.maximum(0.0, r - centers[:, 1])**2)
    pen_bound += np.sum(np.maximum(0.0, r - (1.0 - centers[:, 1]))**2)
    
    # Objective: minimize -r + penalty
    return -r + lam * (pen_overlap + pen_bound)

def run_packing():
    np.random.seed(42)
    n = 26
    r0 = 0.09  # Initial radius guess
    
    # Initialize centers in a staggered hexagonal pattern
    centers = []
    row_counts = [5, 5, 5, 5, 6]  # Sum = 26
    for i, count in enumerate(row_counts):
        y = r0 + i * np.sqrt(3) * r0
        for j in range(count):
            x = r0 + j * 2 * r0
            if i % 2 == 1:
                x += r0
            centers.append([x, y])
    centers = np.array(centers)
    
    # Add small random perturbation to break symmetry and aid convergence
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    
    # Flatten parameters: [x1, y1, ..., x26, y26, r]
    x0 = np.concatenate([centers.ravel(), [r0]])
    bounds = [(0.0, 1.0)] * (2*n) + [(0.01, 0.5)]
    
    # Continuation optimization: gradually increase penalty weight
    current_x = x0
    lam = 500.0
    for _ in range(4):
        res = minimize(compute_objective, current_x, args=(n, lam), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False})
        current_x = res.x
        lam *= 10.0
        
    c_opt = current_x[:2*n].reshape(n, 2)
    r_opt = current_x[2*n]
    
    # Ensure strict boundary compliance
    c_opt = np.clip(c_opt, r_opt, 1.0 - r_opt)
    radii = np.full(n, r_opt)
    
    return c_opt, radii, float(26 * r_opt)
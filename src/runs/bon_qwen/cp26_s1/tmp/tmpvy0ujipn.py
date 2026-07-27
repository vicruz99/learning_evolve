import numpy as np
from scipy.optimize import minimize, linprog
import itertools


def compute_distances(centers, n):
    """Compute pairwise distances between centers."""
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dists[i, j] = d
            dists[j, i] = d
    return dists


def maximize_radii_given_centers(centers, n):
    """Given fixed centers, maximize sum of radii using linear programming."""
    # Variables: r_0, r_1, ..., r_{n-1}
    # Objective: maximize sum(r_i) = -minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Constraints in the form: A_ub @ x <= b_ub
    # r_i >= 0 handled by bounds
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(x)
        # r_i <= 1 - x_i
        A_ub.append(row)
        b_ub.append(1 - x)
        # r_i <= y_i
        A_ub.append(row)
        b_ub.append(y)
        # r_i <= 1 - y_i
        A_ub.append(row)
        b_ub.append(1 - y)
    
    # Non-overlap: r_i + r_j <= dist(i,j)
    dists = compute_distances(centers, n)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        # Fallback: compute radii conservatively
        radii = np.ones(n) * 0.01
        for i in range(n):
            x, y = centers[i]
            r = min(x, 1-x, y, 1-y)
            for j in range(n):
                if i != j:
                    d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    r = min(r, d - radii[j])
            radii[i] = max(0, r)
        return radii


def optimize_positions_given_radii(centers, radii, n):
    """Given fixed radii, optimize positions to spread circles apart."""
    def objective(p):
        c = p.reshape(n, 2)
        # Penalty for overlaps
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c[i] - c[j]) ** 2) + 1e-12)
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    penalty += overlap ** 2
        # Penalty for boundary violations
        for i in range(n):
            x, y = c[i]
            r = radii[i]
            if x - r < 0:
                penalty += (x - r) ** 2
            if x + r > 1:
                penalty += (x + r - 1) ** 2
            if y - r < 0:
                penalty += (y - r) ** 2
            if y + r > 1:
                penalty += (y + r - 1) ** 2
        return penalty
    
    def grad_objective(p):
        c = p.reshape(n, 2)
        g = np.zeros_like(p)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                d = np.sqrt(np.sum(diff ** 2) + 1e-12)
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    factor = -2 * overlap / d
                    g[2*i:2*i+2] += factor * diff
                    g[2*j:2*j+2] -= factor * diff
            
            x, y = c[i]
            r = radii[i]
            if x - r < 0:
                g[2*i] += 2 * (x - r)
            if x + r > 1:
                g[2*i] += 2 * (x + r - 1)
            if y - r < 0:
                g[2*i+1] += 2 * (y - r)
            if y + r > 1:
                g[2*i+1] += 2 * (y + r - 1)
        
        return g
    
    bounds = [(0, 1)] * (2 * n)
    
    result = minimize(objective, centers.flatten(), jac=grad_objective,
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-15})
    
    return result.x.reshape(n, 2)


def hexagonal_init(n, row_config):
    """Initialize centers in a hexagonal pattern."""
    centers = np.zeros((n, 2))
    idx = 0
    
    # Estimate spacing
    total_circles = sum(row_config)
    
    for row, ncols in enumerate(row_config):
        if idx >= n:
            break
        for col in range(ncols):
            if idx >= n:
                break
            x = 0.12 + col * 0.16
            if row % 2 == 1:
                x += 0.08
            y = 0.12 + row * 0.145
            centers[idx] = [x, y]
            idx += 1
    
    return centers


def alternating_optimization(centers, n, max_iters=30):
    """Alternate between optimizing radii and positions."""
    radii = np.ones(n) * 0.05
    best_sum = 0
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    for iteration in range(max_iters):
        # Step 1: Maximize radii given positions (LP)
        radii = maximize_radii_given_centers(centers, n)
        radii = np.clip(radii, 0.001, 0.5)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
        
        # Step 2: Optimize positions given radii
        centers = optimize_positions_given_radii(centers, radii, n)
    
    return best_centers, best_radii, best_sum


def project_to_valid(centers, radii, n):
    """Project solution to satisfy all constraints."""
    # Ensure boundary constraints
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        r = min(r, x, 1-x, y, 1-y)
        r = max(r, 0.001)
        radii[i] = r
    
    # Ensure non-overlap by iteratively reducing radii
    max_passes = 100
    for _ in range(max_passes):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if d < radii[i] + radii[j] - 1e-12:
                    # Reduce the larger radius
                    if radii[i] >= radii[j]:
                        radii[i] = max(0.001, (d - radii[j]) / 2 + 1e-10)
                    else:
                        radii[j] = max(0.001, (d - radii[i]) / 2 + 1e-10)
                    changed = True
        if not changed:
            break
    
    return centers, radii


def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Try multiple row configurations
    row_configs = [
        [5, 4, 5, 4, 5, 3],  # 26 circles
        [4, 5, 4, 5, 4, 4],  # 26 circles  
        [5, 5, 5, 5, 5, 1],  # 26 circles
        [3, 5, 5, 5, 5, 3],  # 26 circles
        [4, 4, 4, 4, 4, 6],  # 26 circles
        [5, 4, 5, 5, 4, 3],  # 26 circles
    ]
    
    for config in row_configs:
        if sum(config) != n:
            continue
        
        centers = hexagonal_init(n, config)
        
        # Alternating optimization
        c_opt, r_opt, s_opt = alternating_optimization(centers, n, max_iters=25)
        
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
    
    # Also try random initialization with multiple restarts
    np.random.seed(42)
    for trial in range(5):
        centers = np.random.rand(n, 2) * 0.8 + 0.1
        radii = np.ones(n) * 0.03
        
        c_opt, r_opt, s_opt = alternating_optimization(centers, n, max_iters=20)
        
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
    
    # Project to valid state
    best_centers, best_radii = project_to_valid(best_centers, best_radii, n)
    
    # Final optimization pass
    best_centers, best_radii, best_sum = alternating_optimization(best_centers, n, max_iters=15)
    best_centers, best_radii = project_to_valid(best_centers, best_radii, n)
    
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum
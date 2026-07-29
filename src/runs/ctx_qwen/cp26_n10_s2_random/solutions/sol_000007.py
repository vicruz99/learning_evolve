# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96e346d6) state=976d5323 sum of radii=1.566596 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_lp_for_radii(centers, A_ub, n, num_pairs):
    """
    Solves the LP to find optimal radii for fixed centers.
    Returns radii, sum_radii, and dual variables.
    """
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)

    # We need to update b_ub in the caller or pass it? 
    # To avoid passing large arrays, let's do it inside or pass pre-allocated.
    # For simplicity in the final structure, I'll construct b_ub here.
    b_ub = np.zeros(num_pairs + 4 * n)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            b_ub[idx] = dists[i, j]
            idx += 1
    
    for i in range(n):
        b_ub[idx] = centers[i, 0]          # x
        idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]   # 1-x
        idx += 1
        b_ub[idx] = centers[i, 1]          # y
        idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]   # 1-y
        idx += 1

    c_obj = -np.ones(n)
    bounds = [(0, None)] * n
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, res.fun * -1, res.ineqlin.marginals
    return None, -np.inf, None

def compute_gradient(centers, duals, n, num_pairs):
    """
    Computes the gradient of the sum of radii with respect to centers.
    """
    grad_centers = np.zeros_like(centers)
    
    # Pairwise terms
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            lam = duals[idx]
            if lam > 1e-9:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d > 1e-9:
                    vec = centers[i] - centers[j]
                    force = (lam / d) * vec
                    grad_centers[i] += force
                    grad_centers[j] -= force
            idx += 1

    # Boundary terms
    boundary_start = num_pairs
    for i in range(n):
        # r_i <= x_i (mu_L)
        mu_L = duals[boundary_start + 4*i]
        # r_i <= 1 - x_i (mu_R)
        mu_R = duals[boundary_start + 4*i + 1]
        # r_i <= y_i (mu_B)
        mu_B = duals[boundary_start + 4*i + 2]
        # r_i <= 1 - y_i (mu_T)
        mu_T = duals[boundary_start + 4*i + 3]
        
        grad_centers[i, 0] += mu_L - mu_R
        grad_centers[i, 1] += mu_B - mu_T
        
    return grad_centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    num_pairs = n * (n - 1) // 2
    total_constraints = num_pairs + 4 * n
    
    # Pre-allocate A_ub
    A_ub = np.zeros((total_constraints, n))
    
    # Fill A_ub
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1
            A_ub[idx, j] = 1
            idx += 1
    
    for i in range(n):
        # r_i <= x_i
        A_ub[idx, i] = 1
        idx += 1
        # r_i <= 1 - x_i
        A_ub[idx, i] = 1
        idx += 1
        # r_i <= y_i
        A_ub[idx, i] = 1
        idx += 1
        # r_i <= 1 - y_i
        A_ub[idx, i] = 1
        idx += 1

    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Generate initial configurations
    configurations = []
    
    # 1. Hexagonal Grid
    r_est = 0.105
    dy = np.sqrt(3)/2 * 2 * r_est
    dx = 2 * r_est
    hex_centers = []
    for row in range(8):
        for col in range(7):
            x = (col + 0.5) * dx + (0.5 if row % 2 != 0 else 0) * dx
            y = (row + 0.5) * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                hex_centers.append([x, y])
        if len(hex_centers) >= n:
            break
            
    if len(hex_centers) >= n:
        base = np.array(hex_centers[:n]) + np.random.normal(0, 0.005, (n, 2))
        base = np.clip(base, 0.02, 0.98)
        configurations.append(base)

    # 2. Random Restarts
    for _ in range(15):
        rand_c = np.random.rand(n, 2) * 0.8 + 0.1
        configurations.append(rand_c)

    for init_centers in configurations:
        centers = init_centers.copy()
        
        for step in range(1000):
            radii, current_sum, duals = solve_lp_for_radii(centers, A_ub, n, num_pairs)
            
            if radii is None:
                break

            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

            grad = compute_gradient(centers, duals, n, num_pairs)
            
            step_size = 0.01 / (1 + 0.002 * step)
            centers += step_size * grad
            centers = np.clip(centers, 0.0, 1.0)
            
            # Check for convergence (optional)
            if np.all(np.abs(grad) < 1e-6):
                break

    return best_centers, best_radii, best_sum

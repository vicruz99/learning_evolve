# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000002 (state 2c120403) state=e2dd90f6 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

N = 26

def get_A_ub():
    """Precompute the inequality constraint matrix A_ub for the LP."""
    num_pairs = N * (N - 1) // 2
    num_boundary = 4 * N
    A = np.zeros((num_pairs + num_boundary, N))
    idx = 0
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i + 1, N):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            idx += 1
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(N):
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
    return A

A_ub = get_A_ub()
bounds_lp = [(0, None)] * N
num_pairs = N * (N - 1) // 2
triu_idx = np.triu_indices(N, k=1)

def solve_lp(centers):
    """Solve LP to find optimal radii for fixed centers. Returns radii, sum, and dual marginals."""
    b_ub = np.zeros(num_pairs + 4 * N)
    
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub[:num_pairs] = dists[triu_idx]
    
    # Vectorized boundary distances
    b_ub[num_pairs:num_pairs+4*N] = np.column_stack([
        centers[:, 0], 1.0 - centers[:, 0],
        centers[:, 1], 1.0 - centers[:, 1]
    ]).flatten()
    
    res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    if res.success:
        return res.x, -res.fun, res.ineqlin.marginals
    return None, -np.inf, None

def compute_gradient(centers, marginals):
    """Compute gradient of the max radii sum w.r.t centers using LP dual variables."""
    grad = np.zeros_like(centers)
    
    # Pairwise gradient contributions
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            lam = marginals[idx]
            if lam > 1e-9:
                d = np.linalg.norm(centers[i] - centers[j])
                if d > 1e-9:
                    vec = (centers[i] - centers[j]) / d
                    grad[i] += lam * vec
                    grad[j] -= lam * vec
            idx += 1
            
    # Boundary gradient contributions
    boundary_start = num_pairs
    for i in range(N):
        mu_L = marginals[boundary_start + 4*i]     # r_i <= x_i
        mu_R = marginals[boundary_start + 4*i + 1] # r_i <= 1-x_i
        mu_B = marginals[boundary_start + 4*i + 2] # r_i <= y_i
        mu_T = marginals[boundary_start + 4*i + 3] # r_i <= 1-y_i
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return grad

def run_gradient_ascent(centers0, max_iter=800):
    """Iteratively optimize centers using LP-derived gradients."""
    centers = centers0.copy()
    best_sum = -1.0
    best_centers = centers.copy()
    best_radii = None
    
    for t in range(max_iter):
        radii, current_sum, marginals = solve_lp(centers)
        if radii is None:
            break
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        grad = compute_gradient(centers, marginals)
        
        # Adaptive step size decay
        step = 0.02 / (1 + 0.005 * t)
        centers += step * grad
        centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
        
    # Final LP solve for the best found configuration
    final_radii, final_sum, _ = solve_lp(best_centers)
    return best_centers, final_radii, final_sum

def get_hex_init(perturb=0.0):
    """Generate hexagonal lattice initialization."""
    centers = np.zeros((N, 2))
    idx = 0
    r = 0.09
    y = r
    row = 0
    while idx < N:
        start_x = r if row % 2 == 0 else 2 * r
        x = start_x
        while x + r <= 1.0 + 1e-9 and idx < N:
            centers[idx, 0] = x + (np.random.rand() - 0.5) * perturb
            centers[idx, 1] = y + (np.random.rand() - 0.5) * perturb
            idx += 1
            x += 2 * r
        y += np.sqrt(3) * r
        row += 1
    return np.clip(centers, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    np.random.seed(42)
    
    # Multiple restarts from diverse initial configurations
    for trial in range(15):
        if trial == 0:
            c0 = get_hex_init(perturb=0.0)
        elif trial == 1:
            c0 = get_hex_init(perturb=0.02)
        else:
            c0 = np.random.rand(N, 2) * 0.8 + 0.1
            
        centers, radii, s = run_gradient_ascent(c0, max_iter=600)
        if s > best_sum:
            best_sum = s
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    # Final repair to guarantee strict validity against numerical tolerance
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(best_centers[i] - best_centers[j])
                if dist < best_radii[i] + best_radii[j] - 1e-12:
                    ov = best_radii[i] + best_radii[j] - dist
                    best_radii[i] -= ov / 2
                    best_radii[j] -= ov / 2
                    changed = True
        for i in range(N):
            x, y = best_centers[i]
            r = min(x, 1 - x, y, 1 - y)
            if best_radii[i] > r + 1e-12:
                best_radii[i] = r
                changed = True
        if not changed:
            break
            
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, np.sum(best_radii)

# sol_000297 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d1ce3e9) state=97da7826 sum of radii=2.617146 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def generate_init(n=26):
    """Generate a hexagonal lattice initial guess with slight jitter."""
    rows = [5, 4, 5, 4, 5, 3]
    centers = []
    y = 0.15
    dy = 0.8 / 5.0
    for count in rows:
        x = 0.15
        dx = 0.7 / (count - 1) if count > 1 else 0.0
        for _ in range(count):
            centers.append([x, y])
            x += dx
        y += dy
        
    centers = np.array(centers[:n])
    # Add small random perturbation to break symmetry
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    # Clip to valid bounds
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(n, 0.04)
    return np.concatenate([centers.flatten(), radii])

def compute_obj(x, n, lam):
    """Objective function: negative sum of radii + penalty for overlaps/boundaries."""
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Inter-circle distances and overlaps
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    radii_sum = radii[:, None] + radii[None, :]
    
    overlap = radii_sum - dists
    np.fill_diagonal(overlap, 0.0)
    # Only consider each pair once
    triu_overlap = np.triu(overlap, k=1)
    pen_inter = np.sum(np.maximum(triu_overlap, 0.0)**2)
    
    # Boundary violations
    pen_bound = np.sum(np.maximum(radii - centers[:, 0], 0.0)**2)
    pen_bound += np.sum(np.maximum(radii - (1.0 - centers[:, 0]), 0.0)**2)
    pen_bound += np.sum(np.maximum(radii - centers[:, 1], 0.0)**2)
    pen_bound += np.sum(np.maximum(radii - (1.0 - centers[:, 1]), 0.0)**2)
    
    return -np.sum(radii) + lam * (pen_inter + pen_bound)

def solve_radii_lp(centers, n):
    """For fixed centers, find optimal radii satisfying all constraints via LP."""
    # Maximize sum(r) <=> minimize -sum(r)
    c_r = np.ones(n) * -1.0
    
    A_ub = []
    b_ub = []
    
    # Pairwise non-overlap constraints: r_i + r_j <= d_ij
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Boundary constraints: r_i <= dist_to_edge
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(centers[i, 0])          # left
        A_ub.append(row.copy())
        b_ub.append(1.0 - centers[i, 0])    # right
        A_ub.append(row.copy())
        b_ub.append(centers[i, 1])          # bottom
        A_ub.append(row.copy())
        b_ub.append(1.0 - centers[i, 1])    # top
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    res = linprog(c_r, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x
    return np.full(n, 0.0)

def run_packing():
    np.random.seed(42)
    n = 26
    x0 = generate_init(n)
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    x_current = x0
    # Annealing schedule for penalty weight
    lams = [10, 50, 200, 1000, 5000]
    
    for lam in lams:
        res = minimize(
            compute_obj, 
            x_current, 
            args=(n, lam), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 1500, 'ftol': 1e-12}
        )
        x_current = res.x
        
    centers = x_current[:2*n].reshape(n, 2)
    # Project to exact valid radii using LP to guarantee feasibility and optimality
    radii = solve_radii_lp(centers, n)
    
    sum_r = np.sum(radii)
    return centers, radii, sum_r

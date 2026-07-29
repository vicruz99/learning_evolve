# sol_000027 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 55285a70) state=fe8f1255 sum of radii=0.994641 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """Solve LP to find maximal radii for fixed centers."""
    n = centers.shape[0]
    # Compute pairwise distances
    d = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    # Compute boundary limits for each circle
    bounds_c = np.minimum(centers, 1.0 - centers).min(axis=1)

    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []

    # Pairwise non-overlap constraints: r_i + r_j <= d_{ij}
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d[i, j])

    # Boundary constraints: r_i <= min distance to edge
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(bounds_c[i])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 1e-6)

def objective_pos(x, n, radii):
    """Objective for position optimization: overlap penalty + repulsion."""
    c = x.reshape(n, 2)
    d = np.linalg.norm(c[:, None] - c[None, :], axis=2)
    
    idx = np.triu_indices(n, k=1)
    d_flat = d[idx]
    r_sum_flat = radii[idx[0]] + radii[idx[1]]

    # Penalty for violations
    overlap = np.maximum(0.0, r_sum_flat - d_flat)
    penalty = np.sum(overlap ** 2)
    
    # Soft repulsion to encourage spreading
    repulsion = np.sum(1.0 / (d_flat + 0.05))
    return penalty + 0.1 * repulsion

def optimize_positions(centers, radii):
    """Optimize center positions given fixed radii."""
    n = centers.shape[0]
    flat_centers = centers.flatten()
    bounds = [(0.0, 1.0)] * (2 * n)
    
    res = minimize(
        objective_pos, 
        flat_centers, 
        args=(n, radii),
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    return res.x.reshape(n, 2)

def run_packing():
    """Main function to run the packing optimization."""
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    for restart in range(10):
        rng = np.random.RandomState(restart * 42 + 7)
        
        if restart == 0:
            # Hexagonal grid initialization
            centers = np.zeros((n, 2))
            idx = 0
            for row in range(7):
                y = 0.12 + row * 0.16
                x = 0.08 if row % 2 == 0 else 0.20
                while x < 0.92 and idx < n:
                    centers[idx] = [x, y]
                    idx += 1
                    x += 0.18
        else:
            # Random initialization with padding
            centers = rng.uniform(0.12, 0.88, size=(n, 2))

        radii = np.full(n, 0.05)

        # Alternating optimization
        for step in range(60):
            radii = solve_radii_lp(centers)
            centers = optimize_positions(centers, radii)
            
        # Final LP step to ensure strict validity for the optimized positions
        radii = solve_radii_lp(centers)
        curr_sum = np.sum(radii)
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, best_sum

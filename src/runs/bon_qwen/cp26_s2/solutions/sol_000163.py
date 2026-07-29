# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abb93ac) state=037e9be3 sum of radii=2.044396 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Perturbed Hexagonal Lattice
    # A hex grid provides a good starting density for circle packing.
    # We generate 26 points, fit them in [0.05, 0.95] to avoid boundary issues initially.
    centers = np.zeros((n, 2))
    idx = 0
    rows = 6
    cols = 5 # 6*5 = 30, we only need 26
    
    # Spacing parameters to fit roughly 26 circles
    spacing_x = 0.20
    spacing_y = spacing_x * math.sqrt(3) / 2
    
    current_y = 0.05
    for r in range(rows):
        current_x = 0.05 + (r % 2) * (spacing_x / 2)
        for c in range(cols):
            if idx < n:
                centers[idx, 0] = current_x + (np.random.rand() - 0.5) * 0.02
                centers[idx, 1] = current_y + (np.random.rand() - 0.5) * 0.02
                idx += 1
            else:
                break
        current_y += spacing_y
    
    # Ensure we are strictly inside the square for the optimizer
    centers = np.clip(centers, 0.001, 0.999)

    # 2. Iterative Optimization Loop
    for iteration in range(300):
        # Compute pairwise distances
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)

        # Compute boundary limits for each circle
        x, y = centers[:, 0], centers[:, 1]
        bounds_radius = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        # Setup LP: Maximize sum(radii) => Minimize -sum(radii)
        c_obj = -np.ones(n)
        m_pairs = n * (n - 1) // 2
        m_total = m_pairs + n
        
        A_ub = np.zeros((m_total, n))
        b_ub = np.zeros(m_total)

        # Pairwise constraints: r_i + r_j <= dist_ij
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A_ub[idx, i] = 1.0
                A_ub[idx, j] = 1.0
                b_ub[idx] = dists[i, j]
                idx += 1

        # Boundary constraints: r_i <= dist_to_boundary
        for i in range(n):
            A_ub[idx, i] = 1.0
            b_ub[idx] = bounds_radius[i]
            idx += 1

        bounds_r = [(0, None) for _ in range(n)]

        # Solve LP
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        
        if res.success:
            radii = res.x
        else:
            # Fallback if LP fails (rare)
            radii = np.zeros(n)

        # Calculate Forces based on tight constraints
        forces = np.zeros((n, 2))
        tol = 1e-4

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                sum_r = radii[i] + radii[j]
                d = dists[i, j]
                if sum_r > d - tol:
                    if d > 1e-9:
                        dir_vec = (centers[i] - centers[j]) / d
                        forces[i] += dir_vec
                        forces[j] -= dir_vec

        # Boundary repulsion (push towards center if tight)
        for i in range(n):
            if radii[i] > bounds_radius[i] - tol:
                # Determine direction to center
                dx = 0.5 - centers[i, 0]
                dy = 0.5 - centers[i, 1]
                mag = np.hypot(dx, dy)
                if mag > 1e-9:
                    forces[i] += (np.array([dx, dy]) / mag)

        # Update centers
        # Decay step size for stability
        step = 0.015 / (1.0 + iteration * 0.05)
        
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        forces = forces / norms
        
        centers += step * forces
        centers = np.clip(centers, 0.001, 0.999)

    # 3. Final Solve
    # Recompute exact distances for final positions
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    x, y = centers[:, 0], centers[:, 1]
    bounds_radius = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

    m_pairs = n * (n - 1) // 2
    m_total = m_pairs + n
    A_ub = np.zeros((m_total, n))
    b_ub = np.zeros(m_total)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
    for i in range(n):
        A_ub[idx, i] = 1.0
        b_ub[idx] = bounds_radius[i]
        idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, method='highs')
    
    if res.success:
        final_radii = res.x
    else:
        final_radii = np.zeros(n)

    sum_radii = float(np.sum(final_radii))
    return centers, final_radii, sum_radii

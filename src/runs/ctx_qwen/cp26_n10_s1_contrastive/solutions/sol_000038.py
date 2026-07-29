# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=9ab2a303 sum of radii=2.622621 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[0::3])

def constraint_func(vars):
    """
    Constraint function: ensures non-overlap between all circle pairs.
    Returns array of constraint values >= 0.
    Boundary constraints are implicitly satisfied by the variable transformation.
    """
    radii = vars[0::3]
    u = vars[1::3]
    v = vars[2::3]
    n = 26
    
    # Transform normalized coordinates to actual positions
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    
    # Compute pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    # Compute squared minimum allowed distances
    r_sum = radii[:, None] + radii[None, :]
    min_dist_sq = r_sum**2
    
    # Extract upper triangular part (i < j) to avoid duplicates and self
    i, j = np.triu_indices(n, k=1)
    return dist_sq[i, j] - min_dist_sq[i, j]

def force_init(n, seed):
    """Generates an initial spread of points using simple force-directed layout."""
    np.random.seed(seed)
    pts = np.random.rand(n, 2) * 0.8 + 0.1
    for _ in range(400):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-4:
                    dist = 1e-4
                    diff = np.random.rand(2) * 1e-4
                forces[i] += diff / (dist**2 + 1e-6)
                forces[j] -= diff / (dist**2 + 1e-6)
            # Wall repulsion to keep points well inside
            for dim in range(2):
                if pts[i, dim] < 0.05: forces[i, dim] += 20.0 * (0.05 - pts[i, dim])
                elif pts[i, dim] > 0.95: forces[i, dim] -= 20.0 * (pts[i, dim] - 0.95)
        pts += forces * 0.005
        pts = np.clip(pts, 0.0, 1.0)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sol = None
    best_val = -np.inf
    
    # Bounds: r in [1e-4, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-4, 0.5), (0, 1), (0, 1)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    inits = []
    
    # 1. Force-directed layouts
    for seed in range(5):
        pts = force_init(n, seed)
        r0 = 0.05
        u0 = (pts[:, 0] - r0) / (1.0 - 2.0 * r0)
        v0 = (pts[:, 1] - r0) / (1.0 - 2.0 * r0)
        vars0 = np.empty(n * 3)
        vars0[0::3] = r0
        vars0[1::3] = u0
        vars0[2::3] = v0
        inits.append(vars0)
        
    # 2. Hexagonal layouts
    for seed in range(5):
        np.random.seed(200 + seed)
        r_hex = 0.085 + seed * 0.002
        positions = []
        y_pos = r_hex
        while len(positions) < n:
            x_pos = r_hex
            while x_pos <= 1.0 - r_hex:
                if len(positions) < n:
                    positions.append((x_pos, y_pos))
                x_pos += 2.0 * r_hex
            y_pos += np.sqrt(3.0) * r_hex
        positions = np.array(positions[:n])
        u_hex = (positions[:, 0] - r_hex) / (1.0 - 2.0 * r_hex)
        v_hex = (positions[:, 1] - r_hex) / (1.0 - 2.0 * r_hex)
        u_hex += np.random.uniform(-0.02, 0.02, n)
        v_hex += np.random.uniform(-0.02, 0.02, n)
        u_hex = np.clip(u_hex, 0.0, 1.0)
        v_hex = np.clip(v_hex, 0.0, 1.0)
        vars0 = np.empty(n * 3)
        vars0[0::3] = r_hex
        vars0[1::3] = u_hex
        vars0[2::3] = v_hex
        inits.append(vars0)
        
    # 3. Grid layout
    np.random.seed(300)
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + i*0.2, 0.1 + j*0.2])
    grid_pts.append([0.5, 0.5])
    grid_pts = np.array(grid_pts)
    r_grid = 0.09
    u_grid = (grid_pts[:, 0] - r_grid) / (1.0 - 2.0 * r_grid)
    v_grid = (grid_pts[:, 1] - r_grid) / (1.0 - 2.0 * r_grid)
    u_grid += np.random.uniform(-0.02, 0.02, n)
    v_grid += np.random.uniform(-0.02, 0.02, n)
    u_grid = np.clip(u_grid, 0.0, 1.0)
    v_grid = np.clip(v_grid, 0.0, 1.0)
    vars0 = np.empty(n * 3)
    vars0[0::3] = r_grid
    vars0[1::3] = u_grid
    vars0[2::3] = v_grid
    inits.append(vars0)

    # Primary optimization loop
    for i, vars0 in enumerate(inits):
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            val = -res.fun
            # Accept only if constraints are satisfied within tolerance
            if np.min(constraint_func(res.x)) >= -1e-6:
                if val > best_val:
                    best_val = val
                    best_sol = res.x
        except Exception:
            continue
            
    # Perturbation refinement on the best solution found
    if best_sol is not None:
        for seed in range(10):
            np.random.seed(400 + seed)
            perturbed = best_sol.copy()
            # Perturb normalized coordinates to escape local minima
            perturbed[1::3] += np.random.uniform(-0.01, 0.01, n)
            perturbed[2::3] += np.random.uniform(-0.01, 0.01, n)
            perturbed[1::3] = np.clip(perturbed[1::3], 0.0, 1.0)
            perturbed[2::3] = np.clip(perturbed[2::3], 0.0, 1.0)
            
            try:
                res = minimize(objective, perturbed, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                val = -res.fun
                if np.min(constraint_func(res.x)) >= -1e-6:
                    if val > best_val:
                        best_val = val
                        best_sol = res.x
            except Exception:
                continue

    # Fallback if optimization fails entirely
    if best_sol is None:
        r_fall = 0.05
        indices = np.arange(n)
        cols = indices % 6
        rows = indices // 6
        x_fall = 0.1 + cols * 0.18
        y_fall = 0.1 + rows * 0.18
        centers = np.column_stack((x_fall, y_fall))
        radii = np.full(n, r_fall)
        return centers, radii, float(np.sum(radii))
        
    # Reconstruct centers from optimized parameters
    radii = best_sol[0::3]
    u = best_sol[1::3]
    v = best_sol[2::3]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    centers = np.column_stack((x, y))
    
    return centers, radii, float(np.sum(radii))

# sol_000121 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000054 (state 94cc489d) state=79bf87f3 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[:N])

def constraints(vars_vec):
    """
    Computes inequality constraints g(vars_vec) >= 0.
    Boundary constraints are handled automatically by the (r, u, v) parameterization.
    Only pairwise non-overlap constraints are enforced.
    """
    r = vars_vec[:N]
    u = vars_vec[N:2*N]
    v = vars_vec[2*N:3*N]
    
    # Parameterization maps u, v in [0,1] to x, y such that circle is strictly inside [0,1]^2
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    return dist_sq[i_idx, j_idx] - r_sum[i_idx, j_idx]**2

def solve_radii_lp(centers):
    """Given fixed centers, solves LP to find optimal feasible radii maximizing sum(r_i)."""
    n = centers.shape[0]
    c = -np.ones(n)  # Minimize -sum(r_i) <=> Maximize sum(r_i)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    return np.full(n, 1e-5)

def make_params(centers, radii):
    """Map physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    d = 1.0 - 2.0 * r
    d = np.clip(d, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / d, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / d, 0.0, 1.0)
    return np.concatenate([r, u, v])

def get_hex_centers(row_counts, rot=0.0, scale=1.0, jitter=0.0):
    """Generate a hexagonal lattice initialization with specified parameters."""
    pts = []
    r_est = 0.1
    y = r_est
    row = 0
    for cnt in row_counts:
        x_start = r_est if row % 2 == 0 else 2.0 * r_est
        for _ in range(cnt):
            pts.append([x_start, y])
            x_start += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    
    # Center and scale
    pts = (pts - 0.5) * scale + 0.5
    
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        
    if jitter > 0:
        pts += np.random.uniform(-jitter, jitter, pts.shape)
        
    return np.clip(pts, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = [(1e-6, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_x = None
    best_sum = -np.inf
    
    # Generate diverse initializations
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5],
        [7,6,6,7], [8,6,6,6], [5,5,5,5,6],
        [6,4,6,5,5], [4,5,6,5,6], [5,7,5,7],
        [6,6,6,6,2], [7,5,6,6,2], [6,5,5,5,5],
        [5,5,6,5,5], [4,4,6,6,6], [6,7,6,7]
    ]
    for pat in patterns:
        for _ in range(4):
            rot = np.random.uniform(-0.3, 0.3)
            scale = np.random.uniform(0.85, 1.15)
            jit = np.random.uniform(0.005, 0.03)
            centers = get_hex_centers(pat, rot=rot, scale=scale, jitter=jit)
            radii = solve_radii_lp(centers) * 0.99
            inits.append(make_params(centers, radii))
            
    # Grid initializations
    for _ in range(10):
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        pts += np.random.uniform(-0.02, 0.02, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        radii = solve_radii_lp(pts) * 0.99
        inits.append(make_params(pts, radii))

    # Phase 1: Broad search from structured initializations
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-14})
            if res.success:
                if np.min(constraints(res.x)) >= -1e-8:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Local perturbation refinement
    if best_x is not None:
        for _ in range(30):
            x0 = best_x.copy()
            x0[:N] += np.random.uniform(-0.002, 0.002, N)
            x0[N:3*N] += np.random.uniform(-0.02, 0.02, 2*N)
            x0[:N] = np.clip(x0[:N], 1e-6, 0.5)
            x0[N:3*N] = np.clip(x0[N:3*N], 0.0, 1.0)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-14})
                if res.success and np.min(constraints(res.x)) >= -1e-8:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
            except Exception:
                pass

    # Phase 3: Alternating center perturbation + LP + SLSQP polish
    if best_x is not None:
        r = best_x[:N]
        u = best_x[N:2*N]
        v = best_x[2*N:3*N]
        cx = r + u * (1.0 - 2.0 * r)
        cy = r + v * (1.0 - 2.0 * r)
        centers = np.column_stack([cx, cy])
        
        for _ in range(25):
            new_centers = centers.copy()
            new_centers += np.random.randn(N, 2) * 0.005
            new_centers = np.clip(new_centers, 0.02, 0.98)
            new_radii = solve_radii_lp(new_centers)
            
            # Check if LP alone improved the sum
            if np.sum(new_radii) > best_sum:
                best_sum = np.sum(new_radii)
                centers = new_centers
                r = new_radii
                d = 1.0 - 2.0 * r
                u = np.clip((centers[:, 0] - r) / d, 0.0, 1.0)
                v = np.clip((centers[:, 1] - r) / d, 0.0, 1.0)
                best_x = np.concatenate([r, u, v])
                
            # Try SLSQP from this improved center configuration
            x0_lp = make_params(new_centers, new_radii * 0.995)
            try:
                res = minimize(objective, x0_lp, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 4000, 'ftol': 1e-13})
                if res.success and np.min(constraints(res.x)) >= -1e-8:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
            except Exception:
                pass

    # Final reconstruction
    r_opt = best_x[:N]
    u_opt = best_x[N:2*N]
    v_opt = best_x[2*N:3*N]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    radii = np.maximum(r_opt, 0.0)
    
    return centers, radii, float(np.sum(radii))

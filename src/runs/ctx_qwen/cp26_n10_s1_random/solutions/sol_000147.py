# sol_000147 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000123 (state 90e3970d) state=286769ef sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(vars_arr):
    """Minimize negative sum of radii to maximize sum of radii."""
    return -np.sum(vars_arr[2 * N :])

def constraints(vars_arr):
    """Compute inequality constraints >= 0 for valid packing."""
    x = vars_arr[:N]
    y = vars_arr[N : 2 * N]
    r = vars_arr[2 * N :]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, None] + r[None, :]
    rs2 = rs**2
    
    i, j = np.triu_indices(N, k=1)
    c_pair = d2[i, j] - rs2[i, j]
    
    return np.concatenate([c_bound, c_pair])

def generate_init(seed, r0=0.085):
    """Generate a perturbed hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    pts = []
    y = r0 + 0.002
    row = 0
    while len(pts) < N + 5:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + 0.002 + shift
        while x + r0 <= 1.0 and len(pts) < N + 5:
            pts.append([x, y])
            x += 2.0 * r0 + 0.004
        y += np.sqrt(3) * r0 + 0.004
        row += 1
    pts = np.array(pts[:N])
    
    # Normalize to fit within [0, 1]
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    pts = (pts - mn) / (mx - mn)
    pts = pts * 0.86 + 0.07
    
    # Add random perturbation
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    pts = np.clip(pts, r0 + 0.01, 1.0 - r0 - 0.01)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Multiple restarts with SLSQP to find optimal centers and radii
    for trial in range(20):
        centers = generate_init(seed=42 + trial, r0=0.085)
        x0 = np.concatenate([centers[:, 0], centers[:, 1], np.full(N, 0.085)])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints},
                options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                x, y, r = res.x[:N], res.x[N : 2 * N], res.x[2 * N :]
                
                # Quick validity check
                valid = True
                if (np.any(x - r < -1e-8) or np.any(1.0 - x - r < -1e-8) or 
                    np.any(y - r < -1e-8) or np.any(1.0 - y - r < -1e-8)):
                    valid = False
                else:
                    d2 = (x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2
                    rs = r[:, None] + r[None, :]
                    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                    if np.any(d2[mask] < rs[mask]**2 - 1e-8):
                        valid = False
                
                if valid:
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = np.column_stack((x, y))
                        best_radii = r.copy()
        except Exception:
            continue

    # Phase 2: LP refinement for radii given the best centers
    if best_centers is not None:
        c_obj = -np.ones(N)
        A_ub = []
        b_ub = []
        bounds_lp = []
        
        for i in range(N):
            bd = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                     best_centers[i, 1], 1.0 - best_centers[i, 1])
            bounds_lp.append((0.0, max(bd, 0.0)))
            
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            lp_res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
            if lp_res.success:
                lp_r = lp_res.x
                valid_lp = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if np.linalg.norm(best_centers[i] - best_centers[j]) < lp_r[i] + lp_r[j] - 1e-10:
                            valid_lp = False
                            break
                    if not valid_lp:
                        break
                if valid_lp:
                    best_radii = lp_r
                    best_sum = np.sum(lp_r)
        except Exception:
            pass
            
        # Phase 3: Strict safety scaling to guarantee numerical validity
        scale = 1.0
        for i in range(N):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            denom = max(r, 1e-12)
            scale = min(scale, (x - r) / denom, (1.0 - x - r) / denom, 
                        (y - r) / denom, (1.0 - y - r) / denom)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                rs = best_radii[i] + best_radii[j]
                if d < rs:
                    scale = min(scale, d / rs)
                    
        scale = max(0.0, scale * 0.9999999)
        best_radii *= scale
        best_sum = float(np.sum(best_radii))

    # Fallback configuration
    if best_centers is None:
        r_fb = 0.08
        grid_pts = [[r_fb + i * 2 * r_fb, r_fb + j * 2 * r_fb] for j in range(5) for i in range(5)]
        best_centers = np.array(grid_pts + [[0.5, 0.5]])
        best_radii = np.full(N, r_fb * 0.99)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum

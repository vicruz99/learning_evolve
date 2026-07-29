# sol_000122 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000054 (state 94cc489d) state=eb428784 sum of radii=2.626815 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(params):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraints(params):
    """
    Computes inequality constraints g(params) >= 0.
    Uses (r, u, v) parameterization so boundary constraints are automatically satisfied.
    Only pairwise non-overlap constraints are enforced.
    """
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Map normalized u, v to actual coordinates within [r, 1-r]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Vectorized pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2

def get_params_from_centers(centers):
    """Maps physical centers to strictly feasible (r, u, v) optimization parameters."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d * 0.5 < mx:
                    mx = d * 0.5
        r[i] = max(1e-5, mx * 0.70)  # Conservative scaling ensures strict feasibility
        
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def gen_hex(row_counts, rot, scale, jitter, seed):
    """Generates a hexagonal lattice configuration with specified parameters."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.095
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
    
    # Center, scale, and rotate
    pts -= 0.5
    pts *= scale
    pts += 0.5
    
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        pts = pts @ np.array([[c, -s], [s, c]])
        
    pts += rng.uniform(-jitter, jitter, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Bounds: r in [1e-6, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_params = None
    best_sum = -np.inf
    rng = np.random.RandomState(42)
    
    inits = []
    
    # Diverse hexagonal patterns summing to >= 26
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 5, 6, 5],
        [7, 6, 6, 7], [8, 6, 6, 6], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 5, 6, 5, 6], [7, 5, 6, 8],
        [5, 7, 5, 6, 3], [6, 6, 6, 5, 3], [4, 4, 6, 6, 6]
    ]
    
    for pat in patterns:
        for s in range(5):
            rot = rng.uniform(-0.35, 0.35)
            scale = rng.uniform(0.85, 1.15)
            jitt = rng.uniform(0.005, 0.03)
            pts = gen_hex(pat, rot, scale, jitt, s)
            inits.append(get_params_from_centers(pts))
            
    # Random strictly feasible starts
    for s in range(30):
        rng = np.random.RandomState(s * 1000)
        pts = rng.rand(N, 2)
        inits.append(get_params_from_centers(pts))
        
    # Phase 1: Broad search from diverse initializations
    for p0 in inits:
        try:
            res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    s_val = np.sum(res.x[:N])
                    if s_val > best_sum:
                        best_sum = s_val
                        best_params = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_params is not None:
        for k in range(40):
            p0 = best_params.copy()
            # Perturb radii slightly
            p0[:N] += rng.uniform(-0.002, 0.002, N)
            # Perturb normalized positions more aggressively to explore topology changes
            p0[N:3*N] += rng.uniform(-0.025, 0.025, 2*N)
            
            p0[:N] = np.clip(p0[:N], 1e-6, 0.5)
            p0[N:3*N] = np.clip(p0[N:3*N], 0.0, 1.0)
            
            try:
                res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                if res.success:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        s_val = np.sum(res.x[:N])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_params = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_params, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res.success and np.min(constraints(res.x)) >= -1e-8:
                best_params = res.x
                best_sum = np.sum(best_params[:N])
        except Exception:
            pass

    # Fallback valid configuration (should rarely be reached)
    if best_params is None:
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        best_params = get_params_from_centers(pts)
        
    # Reconstruct centers and radii
    r_opt = best_params[:N]
    u_opt = best_params[N:2*N]
    v_opt = best_params[2*N:3*N]
    
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    # Final safety scale to guarantee strict compliance with 1e-12 validation tolerance
    radii = np.maximum(r_opt * 0.999999, 0.0)
    
    return centers, radii, float(np.sum(radii))

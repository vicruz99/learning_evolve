# sol_000116 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000105 (state 007a7b0d) state=528be72b sum of radii=2.626995 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(params):
    """Minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraints(params):
    """
    Compute pairwise separation constraints: dist^2 >= (r_i + r_j)^2.
    Boundary constraints are automatically satisfied by the parameterization.
    """
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Recover centers from parameters
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Squared Euclidean distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Extract upper triangular pairs to avoid duplicates and self-comparison
    i_idx, j_idx = np.triu_indices(N, k=1)
    return dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]

def make_params(centers, radii):
    """Map physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    d = 1.0 - 2.0 * r
    d = np.clip(d, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / d, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / d, 0.0, 1.0)
    return np.concatenate([r, u, v])

def get_safe_radii(centers):
    """Compute strictly feasible initial radii for given centers."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                if d * 0.5 < mx:
                    mx = d * 0.5
        r[i] = max(1e-4, mx * 0.85)
    return r

def gen_hex(row_counts, rot, scale, jitter, seed):
    """Generate a hexagonal lattice initialization with specified parameters."""
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
    
    # Center and scale
    pts -= 0.5
    pts *= scale
    pts += 0.5
    
    # Rotate
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        pts -= pts.mean(axis=0)
        pts += 0.5
        
    # Jitter
    pts += rng.uniform(-jitter, jitter, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def gen_force(seed):
    """Spread points using force-directed repulsion within unit square."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    for _ in range(100):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-5
        forces = np.sum(diff / dists[:, :, None] / dists[:, :, None], axis=1)
        pts += 0.02 * forces
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    np.random.seed(42)
    bounds = [(1e-5, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -np.inf
    best_params = None
    
    # Diverse hexagonal patterns covering various boundary alignments
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5],
        [7,6,6,7], [8,6,6,6], [5,5,5,5,6],
        [6,4,6,5,5], [4,5,6,5,6], [5,7,5,7],
        [6,6,6,6,2], [7,5,6,6,2], [6,5,5,5,5],
        [5,5,6,5,5], [4,4,6,6,6], [6,7,6,7], [5,6,6,5,4]
    ]
    
    inits = []
    for pat in patterns:
        for _ in range(6):
            rot = np.random.uniform(-0.45, 0.45)
            scale = np.random.uniform(0.80, 1.25)
            jitt = np.random.uniform(0.01, 0.04)
            inits.append(gen_hex(pat, rot=rot, scale=scale, jitter=jitt, seed=np.random.randint(1000)))
            
    for s in range(15):
        inits.append(gen_force(s))
        
    # Phase 1: Broad search from diverse initializations
    for pts in inits:
        r0 = get_safe_radii(pts)
        p0 = make_params(pts, r0)
        try:
            res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Local perturbation refinement to escape local minima
    if best_params is not None:
        for trial in range(50):
            p0 = best_params.copy()
            p0[:N] += np.random.uniform(-0.0015, 0.0015, N)
            p0[N:3*N] += np.random.uniform(-0.015, 0.015, 2*N)
            p0[:N] = np.clip(p0[:N], 1e-5, 0.5)
            p0[N:3*N] = np.clip(p0[N:3*N], 0.0, 1.0)
            
            try:
                res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                if res.success:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        s_val = -res.fun
                        if s_val > best_sum:
                            best_sum = s_val
                            best_params = res.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_params, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-15, 'disp': False})
            if res.success and np.min(constraints(res.x)) >= -1e-8:
                best_params = res.x
                best_sum = -res.fun
        except Exception:
            pass

    # Fallback configuration (should rarely be reached)
    if best_params is None:
        fallback_centers = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                            np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        fallback_radii = np.full(N, 0.04)
        return fallback_centers, fallback_radii, float(np.sum(fallback_radii))

    # Reconstruct physical centers and radii from optimized parameters
    r = best_params[:N]
    u = best_params[N:2*N]
    v = best_params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    centers = np.column_stack([x, y])
    radii = np.maximum(r, 0.0)
    
    return centers, radii, float(best_sum)

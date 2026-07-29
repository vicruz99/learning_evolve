# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000079 (state c990a719) state=354b144a sum of radii=2.631094 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def obj(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def constr(params):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    return dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def make_params(centers, radii):
    """Map physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    d = 1.0 - 2.0 * r
    d = np.clip(d, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / d, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / d, 0.0, 1.0)
    return np.concatenate([r, u, v])

def hex_init(seed, pattern, scale=1.0, rot=0.0):
    """Generate hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.09
    y = r_est
    row = 0
    for cnt in pattern:
        x_start = r_est if row % 2 == 0 else 2.0 * r_est
        for _ in range(cnt):
            pts.append([x_start, y])
            x_start += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    
    pts -= 0.5
    pts *= scale
    pts += 0.5
    
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        pts -= pts.mean(axis=0)
        pts += 0.5
        
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts

def force_init(seed):
    """Force-directed layout initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    for _ in range(300):
        f = np.zeros_like(pts)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        dist = np.maximum(dist, 1e-4)
        f += np.sum((1.0/dist**2)[:, :, None] * diff / dist[:, :, None], axis=1)
        
        wall = 20.0
        for d in range(2):
            f[:, d] += wall * (pts[:, d] < 0.1) * (0.1 - pts[:, d])
            f[:, d] -= wall * (pts[:, d] > 0.9) * (pts[:, d] - 0.9)
        pts += 0.005 * f
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def safe_radii(centers):
    """Compute strictly feasible initial radii."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                mx = min(mx, d*0.5)
        r[i] = max(1e-4, mx * 0.9)
    return r

def run_packing():
    bounds = [(1e-5, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons = {'type': 'ineq', 'fun': constr}
    
    best_sum = -np.inf
    best_params = None
    
    inits = []
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6], [6,4,6,5,5]]
    for p in patterns:
        for s in range(5):
            inits.append(hex_init(s, p, scale=0.95 + 0.1*(s%3), rot=0.05*(s-2)))
    for s in range(10):
        inits.append(force_init(s))
    for s in range(10):
        rng = np.random.RandomState(s)
        pts = rng.rand(N, 2)
        pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)
        
    for pts in inits:
        r0 = safe_radii(pts)
        p0 = make_params(pts, r0)
        try:
            res = minimize(obj, p0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-13})
            if res.success:
                c_val = constr(res.x)
                if np.min(c_val) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is not None:
        for _ in range(30):
            p0 = best_params.copy()
            p0[N:3*N] += np.random.uniform(-0.008, 0.008, 2*N)
            p0[:N] += np.random.uniform(-0.001, 0.001, N)
            p0[:N] = np.clip(p0[:N], 1e-5, 0.5)
            p0[N:3*N] = np.clip(p0[N:3*N], 0.0, 1.0)
            
            try:
                res = minimize(obj, p0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 2000, 'ftol': 1e-13})
                if res.success:
                    c_val = constr(res.x)
                    if np.min(c_val) >= -1e-7:
                        s_val = -res.fun
                        if s_val > best_sum:
                            best_sum = s_val
                            best_params = res.x.copy()
            except Exception:
                pass
                
        try:
            res = minimize(obj, best_params, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-14})
            if res.success and np.min(constr(res.x)) >= -1e-8:
                best_params = res.x
                best_sum = -res.fun
        except Exception:
            pass

    if best_params is None:
        # Fallback configuration
        fallback_centers = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                            np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        fallback_radii = np.full(N, 0.04)
        return fallback_centers, fallback_radii, float(np.sum(fallback_radii))

    r = best_params[:N]
    u = best_params[N:2*N]
    v = best_params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    centers = np.column_stack([x, y])
    radii = np.maximum(r, 0.0)
    
    return centers, radii, float(best_sum)

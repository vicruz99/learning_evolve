# sol_000072 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=5e97853a sum of radii=2.620433 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, N):
    """Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v, N):
    """Compute inequality constraints: boundary containment and pairwise separation."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise separation: dist(i,j) >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    # Extract upper triangular pairs to avoid duplicates and self-comparison
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c.append(dist[mask] - r_sum[mask])
    
    return np.concatenate(c)

def make_init_from_pts(pts, N, scale_r=0.90):
    """Generate a strictly feasible initial variable vector from center points."""
    r_feas = np.zeros(N)
    for i in range(N):
        d_wall = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        d_min = 1.0
        for j in range(N):
            if i != j:
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < d_min:
                    d_min = d
        # Scale down slightly to guarantee strict feasibility for SLSQP start
        r_feas[i] = scale_r * min(d_wall, 0.5 * d_min)
        
    return np.concatenate([pts[:, 0], pts[:, 1], r_feas])

def run_packing():
    N = 26
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(1e-6, 0.5)] * N
    
    best_sum = -np.inf
    best_v = None
    inits = []
    
    # --- Initialization Generation ---
    
    # 1. Hexagonal lattices with varying target densities
    for r_est in [0.082, 0.092, 0.102]:
        pts = np.zeros((N, 2))
        y = r_est
        row = 0
        idx = 0
        while idx < N:
            x_start = r_est + (row % 2) * r_est
            x = x_start
            while x <= 1.0 - r_est and idx < N:
                pts[idx] = [x, y]
                x += 2.0 * r_est
                idx += 1
            y += np.sqrt(3.0) * r_est
            row += 1
            
        inits.append(make_init_from_pts(pts, N, scale_r=0.92))
        
        # Perturbed hex variants to break symmetry
        rng = np.random.RandomState(100 + int(r_est * 1000))
        pts_p = pts + rng.uniform(-0.018, 0.018, pts.shape)
        pts_p = np.clip(pts_p, 0.02, 0.98)
        inits.append(make_init_from_pts(pts_p, N, scale_r=0.85))
        
    # 2. Square grid patterns
    for sp in [0.13, 0.17, 0.21]:
        pts_g = np.zeros((N, 2))
        idx = 0
        gy = sp
        while idx < N:
            gx = sp
            while gx <= 1.0 - sp and idx < N:
                pts_g[idx] = [gx, gy]
                gx += sp
                idx += 1
            gy += sp
        inits.append(make_init_from_pts(pts_g, N, scale_r=0.88))
        
    # 3. Random feasible starts
    for seed in range(15):
        rng = np.random.RandomState(seed)
        pts_r = rng.uniform(0.05, 0.95, (N, 2))
        inits.append(make_init_from_pts(pts_r, N, scale_r=0.75))
        
    # --- Phase 1: Broad Search ---
    for v0 in inits:
        try:
            res = minimize(
                objective, v0, args=(N,),
                method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'args': (N,)},
                options={'maxiter': 6000, 'ftol': 1e-13}
            )
            if res.success:
                c_val = constraints(res.x, N)
                if np.min(c_val) >= -1e-9:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_v = res.x
        except Exception:
            continue
            
    # --- Phase 2: Local Refinement & Perturbation ---
    if best_v is not None:
        for k in range(30):
            rng = np.random.RandomState(k + 7)
            v_p = best_v.copy()
            v_p += rng.normal(0, 0.0035, v_p.shape)
            
            # Clip to valid bounds
            v_p[:N] = np.clip(v_p[:N], 0.005, 0.995)
            v_p[N:2*N] = np.clip(v_p[N:2*N], 0.005, 0.995)
            v_p[2*N:] = np.maximum(v_p[2*N:], 1e-5)
            
            try:
                res = minimize(
                    objective, v_p, args=(N,),
                    method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraints, 'args': (N,)},
                    options={'maxiter': 4000, 'ftol': 1e-13}
                )
                if res.success:
                    c_val = constraints(res.x, N)
                    if np.min(c_val) >= -1e-9:
                        if -res.fun > best_sum:
                            best_sum = -res.fun
                            best_v = res.x
            except Exception:
                pass
                
    # Fallback (should not be reached)
    if best_v is None:
        best_v = inits[0]
        best_sum = -objective(best_v, N)
        
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(best_sum)

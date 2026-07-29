# sol_000185 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000143 (state c665f9c9) state=e14b5318 sum of radii=2.622893 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRI_U = np.triu_indices(N, k=1)

def get_lp_matrices(n):
    """Pre-construct the sparse structure of the LP constraint matrix."""
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A_ub = np.zeros((num_pairs + num_bound, n))
    pair_indices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pair_indices.append((i, j))
            idx += 1
    for i in range(n):
        for _ in range(4):
            A_ub[idx, i] = 1.0
            idx += 1
    return A_ub, pair_indices

A_L, P_IDX = get_lp_matrices(N)

def solve_lp_and_grad(centers):
    """
    Solves LP for optimal radii given fixed centers and computes the gradient 
    of the sum of radii with respect to center positions using LP duals.
    """
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(A_L.shape[0])
    idx = 0
    for i, j in P_IDX:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    bounds_r = [(0, u) for u in ub]
    
    res = linprog(c_obj, A_ub=A_L, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return None, -1.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Safely extract duals across different scipy versions
    if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
        duals = res.ineqlin.marginals
    elif hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
        duals = res.marginals.ineqlin
    else:
        duals = np.zeros_like(b_ub)
        
    grad = np.zeros_like(centers)
    
    idx = 0
    for i, j in P_IDX:
        lam = duals[idx]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    b_start = len(P_IDX)
    for i in range(n):
        mu_L = duals[b_start + 4*i]
        mu_R = duals[b_start + 4*i + 1]
        mu_B = duals[b_start + 4*i + 2]
        mu_T = duals[b_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, np.sum(radii), grad

def generate_inits(rng, n):
    """Generates diverse initial configurations."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4], [5, 5, 5, 5, 6],
        [7, 5, 5, 5, 4], [5, 7, 5, 5, 4], [6, 4, 6, 5, 5],
        [5, 5, 5, 6, 5], [5, 6, 5, 5, 5], [4, 5, 6, 6, 5],
        [6, 5, 5, 5, 5], [5, 5, 5, 5, 6], [4, 6, 5, 6, 5]
    ]
    for pat in patterns:
        for r0 in [0.09, 0.095, 0.10, 0.105, 0.11]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3)
            c = np.array(c[:n])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.05, 0.95)
            inits.append(c)
            
    # Force-directed random starts to explore non-lattice optima
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (n, 2))
        for _ in range(300):
            for i in range(n):
                for j in range(i+1, n):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.25 and dist > 1e-6:
                        push = (0.25 - dist) * 0.04
                        c[i] += d_vec / dist * push
                        c[j] -= d_vec / dist * push
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    return inits

def run_packing():
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_inits(rng, N)
    
    # Phase 1: LP Gradient Ascent
    for c0 in starts:
        c = c0.copy()
        step = 0.004
        for k in range(1500):
            radii, s, grad = solve_lp_and_grad(c)
            if radii is None: break
            if s > best_sum:
                best_sum = s
                best_c = c.copy()
                best_r = radii.copy()
                
            gn = np.linalg.norm(grad)
            if gn < 1e-10: break
            
            c += step * (grad / gn)
            c = np.clip(c, 0.001, 0.999)
            
            if k % 60 == 0 and k > 0: step *= 0.96
            if k % 200 == 0: 
                c += rng.normal(0, 0.0015 * (0.5**(k//200)), c.shape)
                c = np.clip(c, 0.02, 0.98)
                
    # Phase 2: SLSQP Joint Polish
    if best_c is not None:
        bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
        
        def obj_sl(v):
            return -np.sum(v[2*N:])
            
        def cons_sl(v):
            cc = v[:2*N].reshape(N, 2)
            rr = v[2*N:]
            con = [cc[:,0]-rr, 1.0-cc[:,0]-rr, cc[:,1]-rr, 1.0-cc[:,1]-rr]
            idx = TRI_U
            d = np.linalg.norm(cc[idx[0]] - cc[idx[1]], axis=1)
            con.append(d - (rr[idx[0]] + rr[idx[1]]))
            return np.concatenate(con)
            
        v0 = np.concatenate([best_c.flatten(), best_r])
        
        for _ in range(8):
            v_try = v0.copy()
            v_try[:2*N] += rng.normal(0, 0.0008, 2*N)
            v_try[:2*N] = np.clip(v_try[:2*N], 0.01, 0.99)
            try:
                res = minimize(obj_sl, v_try, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': cons_sl},
                              options={'maxiter': 6000, 'ftol': 1e-14})
                if np.min(cons_sl(res.x)) >= -1e-7:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
                        v0 = res.x.copy()
            except Exception: pass

    centers = best_c.copy()
    radii = best_r.copy()
    
    # Phase 3: Strict Deterministic Repair
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d)/2.0 + 1e-12
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr - 1e-12:
                radii[i] = mr
                changed = True
        if not changed: break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))

# sol_000161 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000133 (state 27fd9551) state=3495d006 sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute LP constraint matrix structure (constant for fixed N)
PAIR_INDICES = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        PAIR_INDICES.append((i, j))
N_PAIRS = len(PAIR_INDICES)
A_ub_pre = np.zeros((N_PAIRS, N))
for k, (i, j) in enumerate(PAIR_INDICES):
    A_ub_pre[k, i] = 1.0
    A_ub_pre[k, j] = 1.0

def solve_lp_radii(centers):
    """Solves LP to find maximum sum of radii for fixed centers."""
    c = np.clip(centers, 1e-9, 1.0 - 1e-9)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-7)
    
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(N_PAIRS)
    for k, (i, j) in enumerate(PAIR_INDICES):
        b_ub[k] = dists[i, j]
        
    bounds = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_ub_pre, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, -res.fun
    return np.zeros(N), 0.0

def lp_obj(x):
    """Objective for centers-only optimization: minimize negative LP sum of radii."""
    _, s = solve_lp_radii(x.reshape(N, 2))
    return -s

def generate_starts():
    """Generates diverse initial configurations."""
    starts = []
    rng = np.random.default_rng(42)
    
    # 1. Hexagonal lattice patterns with varying row structures
    patterns = [[5,5,5,5,6], [5,6,5,6,4], [6,5,6,5,4], [4,6,6,6,4], [5,5,5,6,5], 
                [6,6,5,5,4], [5,4,6,6,5], [6,5,5,5,5], [5,5,4,6,6], [4,5,6,5,6]]
    for pat in patterns:
        for r0 in [0.09, 0.095, 0.10, 0.105]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3)
            starts.append(np.array(c[:N]))
            
    # 2. Grid patterns
    for sp in [5, 6]:
        x = np.linspace(0.1, 0.9, sp)
        y = x.copy()
        cx, cy = np.meshgrid(x, y)
        g = np.column_stack([cx.flatten(), cy.flatten()])
        if len(g) > N:
            g = g[:N]
        starts.append(g)
        
    # 3. Random starts
    for _ in range(8):
        starts.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return starts

def obj_joint(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Computes boundary and non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
    idx_i, idx_j = np.triu_indices(N, 1)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    dr = r[idx_i] + r[idx_j]
    cons.append(np.sqrt(dx**2 + dy**2) - dr)
    return np.concatenate(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(123)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    bounds_j = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    bounds_c = [(0.0, 1.0)] * (2*N)
    
    starts = generate_starts()
    
    # --- Phase 1: SLSQP Joint Optimization ---
    for c_init in starts:
        rb = np.minimum(np.minimum(c_init[:,0], 1.0-c_init[:,0]), np.minimum(c_init[:,1], 1.0-c_init[:,1]))
        dists = np.linalg.norm(c_init[:,None,:] - c_init[None,:, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(rb, rp) * 0.85
        
        v0 = np.concatenate([c_init.flatten(), r_init])
        for _ in range(2):
            v_curr = v0 + rng.normal(0, 0.003, v0.shape)
            v_curr = np.clip(v_curr, 0.01, 0.99)
            v_curr[2*N:] = np.clip(v_curr[2*N:], 0.01, 0.4)
            try:
                res = minimize(obj_joint, v_curr, method='SLSQP', bounds=bounds_j,
                              constraints={'type': 'ineq', 'fun': cons_joint},
                              options={'maxiter': 5000, 'ftol': 1e-12})
                if np.min(cons_joint(res.x)) >= -1e-9:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
            except Exception:
                pass

    # --- Phase 2: Powell on Centers with LP Objective ---
    if best_c is not None:
        res_p = minimize(lp_obj, best_c.flatten(), method='Powell', bounds=bounds_c,
                        options={'maxiter': 1000, 'ftol': 1e-12})
        c_p = res_p.x.reshape(N, 2)
        r_p, s_p = solve_lp_radii(c_p)
        if s_p > best_sum:
            best_sum = s_p
            best_c = c_p
            best_r = r_p
            
        for _ in range(4):
            c_tr = best_c + rng.normal(0, 0.005, best_c.shape)
            c_tr = np.clip(c_tr, 0.02, 0.98)
            res_p2 = minimize(lp_obj, c_tr.flatten(), method='Powell', bounds=bounds_c,
                             options={'maxiter': 800, 'ftol': 1e-12})
            c2 = res_p2.x.reshape(N, 2)
            r2, s2 = solve_lp_radii(c2)
            if s2 > best_sum:
                best_sum = s2
                best_c = c2
                best_r = r2

    # --- Phase 3: Simulated Annealing on Centers ---
    c_curr = best_c.copy()
    s_curr = best_sum
    r_curr = best_r.copy()
    T = 0.008
    for step in range(1200):
        T *= 0.9996
        c_new = c_curr + rng.normal(0, T, c_curr.shape)
        c_new = np.clip(c_new, 0.02, 0.98)
        r_new, s_new = solve_lp_radii(c_new)
        
        if s_new > s_curr:
            c_curr = c_new
            s_curr = s_new
            r_curr = r_new
            if s_curr > best_sum:
                best_sum = s_curr
                best_c = c_curr.copy()
                best_r = r_curr.copy()
        else:
            if rng.random() < np.exp((s_new - s_curr) / max(T * 5.0, 1e-7)):
                c_curr = c_new
                s_curr = s_new
                r_curr = r_new

    # --- Phase 4: Final SLSQP Polish ---
    r_lp, _ = solve_lp_radii(best_c)
    v0 = np.concatenate([best_c.flatten(), r_lp])
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_j,
                      constraints={'type': 'ineq', 'fun': cons_joint},
                      options={'maxiter': 5000, 'ftol': 1e-12})
        if np.min(cons_joint(res.x)) >= -1e-9:
            s = np.sum(res.x[2*N:])
            if s > best_sum:
                best_sum = s
                best_c = res.x[:2*N].reshape(N, 2).copy()
                best_r = res.x[2*N:].copy()
    except Exception:
        pass

    # --- Phase 5: Strict Numerical Repair ---
    centers = best_c.copy()
    radii = best_r.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))

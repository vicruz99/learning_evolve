# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000111 (state 4b754d5d) state=720e5d00 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_max_radii(centers):
    """Computes the maximum feasible radius for each circle given fixed centers."""
    c = np.clip(centers, 1e-9, 1.0 - 1e-9)
    rb = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    dists = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    rp = 0.5 * np.min(dists, axis=1)
    return np.minimum(rb, rp)

def obj_func(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints_func(v):
    """Computes boundary and non-overlap constraints. Must be >= 0."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    # Boundary constraints
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    # Overlap constraints
    idx = np.triu_indices(N, 1)
    d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
    con.append(d - (r[idx[0]] + r[idx[1]]))
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(2024)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    best_sum = -1.0
    best_c = None
    best_r = None
    
    starts = []
    
    # 1. Corner & Edge biased configurations
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        starts.append(c)
        
    # 2. Hexagonal patterns with variations
    for r0 in [0.09, 0.10, 0.105, 0.11]:
        c = []
        y = r0
        row = 0
        while len(c) < N:
            x = r0 if row % 2 == 0 else 2 * r0
            while x + r0 <= 1.0 and len(c) < N:
                c.append([x, y])
                x += 2 * r0
            y += r0 * np.sqrt(3)
            row += 1
        c = np.array(c[:N]) + rng.normal(0, 0.003, (N, 2))
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # 3. Grid patterns
    for s in [5, 6]:
        pts = []
        step = 0.8 / (s - 1) if s > 1 else 0.5
        for i in range(s):
            for j in range(s):
                if len(pts) < N:
                    pts.append([0.1 + i * step, 0.1 + j * step])
        starts.append(np.array(pts[:N]))
        
    # 4. Vectorized Growth Simulation
    for seed in range(4):
        rng_local = np.random.default_rng(seed)
        c = rng_local.uniform(0.2, 0.8, (N, 2))
        r = np.full(N, 0.03)
        for _ in range(800):
            r *= 1.0003
            diff = c[:, None, :] - c[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            overlap = np.maximum(0, r[:, None] + r[None, :] - dists)
            dirs = diff / (dists[:, :, None] + 1e-9)
            forces = np.sum(dirs * overlap[:, :, None], axis=1)
            # Vectorized boundary repulsion
            forces[:, 0] += np.maximum(0, r - c[:, 0]) * 10.0 - np.maximum(0, c[:, 0] - (1.0 - r)) * 10.0
            forces[:, 1] += np.maximum(0, r - c[:, 1]) * 10.0 - np.maximum(0, c[:, 1] - (1.0 - r)) * 10.0
            c += 0.5 * forces * 0.001
            c = np.clip(c, 0.0, 1.0)
        starts.append(c)
        
    # Phase 1: Primary Optimization
    for c0 in starts:
        r0 = compute_max_radii(c0)
        v0 = np.concatenate([c0.flatten(), r0])
        try:
            res = minimize(obj_func, v0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12})
            if np.min(constraints_func(res.x)) >= -1e-9:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation & Symmetry Breaking
    if best_c is not None:
        for i in range(25):
            noise = 0.006 * (0.85 ** (i // 3))
            c_p = best_c.copy() + rng.normal(0, noise, best_c.shape)
            c_p = np.clip(c_p, 0.02, 0.98)
            
            # Periodic swap to break identical circle symmetries
            if i % 3 == 0:
                idx = rng.choice(N, 2, replace=False)
                c_p[idx] = c_p[idx[::-1]]
                
            r_p = compute_max_radii(c_p) * 0.99
            v_p = np.concatenate([c_p.flatten(), r_p])
            try:
                res = minimize(obj_func, v_p, method='SLSQP', bounds=bounds, 
                              constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12})
                if np.min(constraints_func(res.x)) >= -1e-9:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
            except Exception:
                pass
                
    # Phase 3: Strict Numerical Repair
    centers = best_c.copy()
    radii = best_r.copy()
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < radii[i] + radii[j] - 1e-12:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                        centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))

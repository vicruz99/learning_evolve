# sol_000225 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000123 (state 101aee21) state=919d6f92 sum of radii=1.869867 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def compute_constraints(v):
    """Computes boundary and non-overlap constraints. Must return >= 0."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    X = x[:, None] - x[None, :]
    Y = y[:, None] - y[None, :]
    R = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c = np.concatenate([c, (X**2 + Y**2)[mask] - R[mask]**2])
    return c

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def force_directed_init(rng, iters=1500):
    """Generates initial configuration using repulsive forces."""
    pts = rng.uniform(0.2, 0.8, (N, 2))
    for _ in range(iters):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 0.15 and dist > 1e-5:
                    rep = (0.15 - dist) * 0.12
                    f = rep * diff / dist
                    forces[i] += f
                    forces[j] -= f
            # Boundary repulsion
            for k in range(2):
                if pts[i, k] < 0.1: forces[i, k] += (0.1 - pts[i, k]) * 2.0
                elif pts[i, k] > 0.9: forces[i, k] -= (pts[i, k] - 0.9) * 2.0
        pts += forces
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def hex_init(pat, r0, rng):
    """Generates hexagonal lattice pattern with given row counts."""
    pts = []
    y = r0
    for idx, cnt in enumerate(pat):
        shift = r0 if idx % 2 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) < N:
                pts.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
            x += 2.0 * r0
        y += r0 * math.sqrt(3)
    while len(pts) < N:
        pts.append(rng.uniform(0.2, 0.8, 2))
    return np.array(pts[:N])

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    num_cons = N * (N - 1) // 2 + 4 * N
    A = np.zeros((num_cons, N))
    b = np.zeros(num_cons)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[k, i] = 1.0
            A[k, j] = 1.0
            b[k] = np.linalg.norm(centers[i] - centers[j])
            k += 1
    for i in range(N):
        x, y = centers[i]
        A[k, i] = 1.0; b[k] = x; k += 1
        A[k, i] = 1.0; b[k] = 1.0 - x; k += 1
        A[k, i] = 1.0; b[k] = y; k += 1
        A[k, i] = 1.0; b[k] = 1.0 - y; k += 1
        
    res = linprog(-np.ones(N), A_ub=A, b_ub=b, bounds=[(0, 0.5)] * N, method='highs')
    if res.success:
        return res.x
    return np.full(N, 0.01)

def obj_lp(v):
    """Objective for center optimization: maximize LP-derived sum of radii."""
    c = v.reshape(N, 2)
    r = solve_lp_radii(c)
    return -np.sum(r)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_v = None
    best_sum = -np.inf
    
    candidates = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,5,5,6], [6,6,5,5,4], 
        [4,6,6,6,4], [5,5,6,5,5], [6,4,6,5,5], [5,5,4,6,6]
    ]
    for pat in patterns:
        for r0 in [0.095, 0.10, 0.105, 0.11]:
            candidates.append(hex_init(pat, r0, rng))
            
    for _ in range(15):
        candidates.append(force_directed_init(rng))
        
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        candidates.append(c)

    # Phase 1: Primary SLSQP Optimization
    for c0 in candidates:
        rb = np.min(np.stack([c0[:,0], 1.0-c0[:,0], c0[:,1], 1.0-c0[:,1]], axis=1), axis=1)
        dists = np.linalg.norm(c0[:, None] - c0[None, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rn = 0.5 * np.min(dists, axis=1)
        r0 = np.minimum(rb, rn) * 0.96
        
        v0 = np.zeros(3 * N)
        v0[0::3] = c0[:, 0]
        v0[1::3] = c0[:, 1]
        v0[2::3] = r0
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if np.min(compute_constraints(res.x)) >= -1e-7:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = candidates[0].flatten()
        best_v[2::3] = 0.05
        
    # Phase 2: Basin Hopping & Perturbation
    curr_v = best_v.copy()
    for step in range(120):
        noise = 0.012 * (0.95 ** (step // 4))
        v_pert = curr_v + rng.normal(0, noise, 3 * N)
        v_pert[0::3] = np.clip(v_pert[0::3], 0.02, 0.98)
        v_pert[1::3] = np.clip(v_pert[1::3], 0.02, 0.98)
        v_pert[2::3] = np.clip(v_pert[2::3], 0.01, 0.45)
        
        # Symmetry breaking swap
        if step % 12 == 0:
            idx = rng.choice(N, 2, replace=False)
            v_pert[3*idx] = v_pert[3*idx[::-1]]
            v_pert[3*idx+1] = v_pert[3*idx+1][::-1]
            v_pert[3*idx+2] = v_pert[3*idx+2][::-1]
            
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
            if np.min(compute_constraints(res.x)) >= -1e-7:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
                    curr_v = best_v.copy()
        except Exception:
            continue
            
    # Phase 3: LP Radius Polish
    centers = best_v[:2*N].reshape(N, 2)
    radii = solve_lp_radii(centers)
    best_sum = np.sum(radii)
    
    # Phase 4: Powell Center Refinement
    c_flat = centers.flatten()
    try:
        res_p = minimize(obj_lp, c_flat, method='Powell', bounds=[(0,1)]*(2*N), 
                        options={'maxiter': 1200, 'ftol': 1e-12})
        c_opt = res_p.x.reshape(N, 2)
        r_opt = solve_lp_radii(c_opt)
        if np.sum(r_opt) > best_sum:
            centers = c_opt
            radii = r_opt
            best_sum = np.sum(r_opt)
    except Exception:
        pass
        
    # Phase 5: Final Joint SLSQP Polish
    v_final = np.concatenate([centers.flatten(), radii])
    try:
        res_f = minimize(objective, v_final, method='SLSQP', bounds=bounds,
                        constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14})
        if np.min(compute_constraints(res_f.x)) >= -1e-7:
            s_f = -res_f.fun
            if s_f > best_sum:
                centers = res_f.x[:2*N].reshape(N, 2)
                radii = res_f.x[2*N:]
                best_sum = s_f
    except Exception:
        pass
        
    # Phase 6: Strict Numerical Repair
    for _ in range(50):
        changed = False
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))

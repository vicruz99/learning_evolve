# sol_000360 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000320 (state 24d66f03) state=f30435ab sum of radii=2.630751 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure (constant across runs)
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_IDX.append((i, j))
        idx += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii and computes exact subgradient via duals."""
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-10:
            d = dists[i, j]
            if d > 1e-10:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def obj_lp_centers(x):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    centers = x.reshape(N, 2)
    _, s, g = solve_lp_and_grad(centers)
    return -s, -g.flatten()

def constraints_joint(v):
    """Computes boundary and non-overlap constraints for joint SLSQP optimization."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def objective_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def force_init(rng):
    """Generates a well-spaced configuration via repulsive forces."""
    c = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(600):
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i+1, N):
                dv = c[i] - c[j]
                d = np.linalg.norm(dv)
                if d < 0.22 and d > 1e-4:
                    push = (0.22 - d) * 0.04 / (d + 1e-4)
                    f[i] += dv / d * push
                    f[j] -= dv / d * push
        c += f * 0.4
        c = np.clip(c, 0.05, 0.95)
    return c

def corner_edge_init(rng):
    """Places circles strategically in corners and along edges."""
    c = np.zeros((N, 2))
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    c[:4] = corners
    # Edge placements
    edge_pts = []
    for _ in range(4):
        edge_pts.append([rng.uniform(0.15, 0.85), 0.08])
        edge_pts.append([rng.uniform(0.15, 0.85), 0.92])
        edge_pts.append([0.08, rng.uniform(0.15, 0.85)])
        edge_pts.append([0.92, rng.uniform(0.15, 0.85)])
    c[4:20] = np.array(edge_pts)
    # Remaining in center
    c[20:] = rng.uniform(0.3, 0.7, (6, 2))
    c += rng.normal(0, 0.005, c.shape)
    return np.clip(c, 0.02, 0.98)

def coordinate_descent(centers, rng, cycles=3):
    """Optimizes each circle's position independently using derivative-free methods."""
    c = centers.copy()
    for _ in range(cycles):
        for i in range(N):
            def local_obj(x):
                temp = c.copy()
                temp[i] = np.clip(x, 1e-5, 1.0 - 1e-5)
                _, s, _ = solve_lp_and_grad(temp)
                return -s
            try:
                res = minimize(local_obj, c[i], method='Powell', 
                               options={'maxiter': 300, 'xtol': 1e-8})
                if res.success:
                    c[i] = np.clip(res.x, 1e-5, 1.0 - 1e-5)
            except Exception:
                pass
    return c

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing():
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    # Generate diverse initial configurations
    starts = []
    
    # Hexagonal patterns
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5]]
    for pat in pats:
        for r0 in [0.090, 0.096, 0.102, 0.108]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    for _ in range(15):
        starts.append(force_init(rng))
    for _ in range(10):
        starts.append(corner_edge_init(rng))
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))

    bounds_c = [(0.001, 0.999)] * (2 * N)
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # Phase 1: Multi-start L-BFGS-B
    for c_init in starts:
        try:
            res1 = minimize(obj_lp_centers, c_init.flatten(), jac=True, method='L-BFGS-B', 
                            bounds=bounds_c, options={'maxiter': 5000, 'ftol': 1e-15})
            c1 = res1.x.reshape(N, 2)
            r1, s1, _ = solve_lp_and_grad(c1)
            if s1 > best_s:
                best_s = s1
                best_c = c1.copy()
                best_r = r1.copy()
        except Exception:
            pass

    # Phase 2: Joint SLSQP on best and diverse starts
    candidates = [best_c] + starts[:10]
    for c_init in candidates:
        try:
            ub = np.minimum(np.minimum(c_init[:, 0], 1.0 - c_init[:, 0]), 
                            np.minimum(c_init[:, 1], 1.0 - c_init[:, 1]))
            dists = np.linalg.norm(c_init[:, None, :] - c_init[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            rp = 0.5 * np.min(dists, axis=1)
            r0_init = np.minimum(ub, rp) * 0.88
            
            v0 = np.concatenate([c_init.flatten(), r0_init])
            res2 = minimize(objective_joint, v0, method='SLSQP', bounds=bounds_joint,
                            constraints={'type': 'ineq', 'fun': constraints_joint},
                            options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraints_joint(res2.x)) >= -1e-7:
                s2 = np.sum(res2.x[2*N:])
                if s2 > best_s:
                    best_s = s2
                    best_c = res2.x[:2*N].reshape(N, 2).copy()
                    best_r = res2.x[2*N:].copy()
        except Exception:
            pass

    # Phase 3: Targeted Kicks & Coordinate Descent
    for _ in range(30):
        c_k = best_c.copy()
        idx = rng.choice(N, size=rng.integers(3, 8), replace=False)
        c_k[idx] += rng.normal(0, 0.015, (len(idx), 2))
        c_k = np.clip(c_k, 0.05, 0.95)
        
        try:
            res_k = minimize(obj_lp_centers, c_k.flatten(), jac=True, method='L-BFGS-B', 
                             bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-14})
            c_k2 = res_k.x.reshape(N, 2)
            r_k2, s_k2, _ = solve_lp_and_grad(c_k2)
            if s_k2 > best_s:
                best_s = s_k2
                best_c = c_k2.copy()
                best_r = r_k2.copy()
                
                # Follow up with coordinate descent
                c_cd = coordinate_descent(best_c, rng, cycles=2)
                r_cd, s_cd, _ = solve_lp_and_grad(c_cd)
                if s_cd > best_s:
                    best_s = s_cd
                    best_c = c_cd.copy()
                    best_r = r_cd.copy()
        except Exception:
            pass

    # Phase 4: Simulated Annealing
    c_sa = best_c.copy()
    s_sa = best_s
    T = 0.008
    for step in range(1200):
        noise_scale = T * 0.5
        c_try = c_sa + rng.normal(0, noise_scale, c_sa.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_sa
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            c_sa, s_sa = c_try, s_try
            if s_sa > best_s:
                best_s = s_sa
                best_c = c_sa.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.996

    # Phase 5: Micro-optimization / Gradient Polish with momentum
    c_pol = best_c.copy()
    v_pol = np.zeros_like(c_pol)
    step_size = 0.003
    for _ in range(1500):
        _, _, grad = solve_lp_and_grad(c_pol)
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            direction = grad / g_norm
            v_pol = 0.8 * v_pol + 0.2 * step_size * direction
            c_new = np.clip(c_pol + v_pol, 0.001, 0.999)
            _, s_new, _ = solve_lp_and_grad(c_new)
            
            if s_new > best_s:
                c_pol = c_new
                best_s = s_new
                best_c = c_pol.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
                step_size = min(step_size * 1.05, 0.008)
            else:
                step_size *= 0.95
        else:
            step_size *= 0.95
            
    # Phase 6: Final Joint Polish
    try:
        v0_final = np.concatenate([best_c.flatten(), best_r])
        res_final = minimize(objective_joint, v0_final, method='SLSQP', bounds=bounds_joint,
                             constraints={'type': 'ineq', 'fun': constraints_joint},
                             options={'maxiter': 15000, 'ftol': 1e-15, 'disp': False})
        if np.min(constraints_joint(res_final.x)) >= -1e-8:
            s_final = np.sum(res_final.x[2*N:])
            if s_final > best_s:
                best_c = res_final.x[:2*N].reshape(N, 2)
                best_r = res_final.x[2*N:]
                best_s = s_final
    except Exception:
        pass
        
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))

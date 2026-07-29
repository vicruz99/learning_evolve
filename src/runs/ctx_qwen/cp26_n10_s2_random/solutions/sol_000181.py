# sol_000181 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000155 (state 1d52432b) state=6225129a sum of radii=2.621304 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute LP constraint matrix structure
def build_lp_structure(n):
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A = np.zeros((num_pairs + num_bound, n))
    pairs = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            pairs.append((i, j))
            idx += 1
    for i in range(n):
        for _ in range(4):
            A[idx, i] = 1.0
            idx += 1
    return A, pairs

LP_A, LP_PAIRS = build_lp_structure(N)
NUM_PAIRS = len(LP_PAIRS)

def solve_lp_radii(centers):
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-10)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(LP_A.shape[0])
    idx = 0
    for i, j in LP_PAIRS:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
    
    res = linprog(-np.ones(n), A_ub=LP_A, b_ub=b_ub,
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        return res.x, -res.fun
    return np.full(n, 0.05), 0.0

def lp_obj_and_grad(v):
    centers = v.reshape(N, 2)
    centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
    
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-10)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    b_ub = np.zeros(LP_A.shape[0])
    idx = 0
    for i, j in LP_PAIRS:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
    
    res = linprog(-np.ones(N), A_ub=LP_A, b_ub=b_ub,
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return 0.0, np.zeros_like(v)
    
    duals = res.ineqlin.marginals
    sum_r = -res.fun
    
    grad = np.zeros((N, 2))
    idx = 0
    for i, j in LP_PAIRS:
        lam = duals[idx]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-10:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
    
    boundary_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[boundary_start + 4*i] - duals[boundary_start + 4*i + 1]
        grad[i, 1] += duals[boundary_start + 4*i + 2] - duals[boundary_start + 4*i + 3]
    
    return -sum_r, -grad.reshape(-1)

def obj_func_centers(v):
    centers = v.reshape(N, 2)
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    _, s = solve_lp_radii(centers)
    return -s

def generate_hex_config(pat, r_est, noise_rng, noise_scale=0.003):
    c = []
    y = r_est
    for r_idx, cnt in enumerate(pat):
        shift = r_est if r_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            c.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3)
    c = np.array(c[:N])
    c += noise_rng.normal(0, noise_scale, c.shape)
    c = np.clip(c, 0.05, 0.95)
    return c

def generate_corner_config(rng):
    c = rng.uniform(0.15, 0.85, (N, 2))
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for i, corner in enumerate(corners):
        c[i] = corner
    c += rng.normal(0, 0.01, c.shape)
    c = np.clip(c, 0.02, 0.98)
    return c

def generate_rotated_hex(rng, base_r=0.095):
    pts = []
    for i in range(-5, 10):
        for j in range(-5, 10):
            px = i * base_r + (j % 2) * 0.5 * base_r
            py = j * base_r * np.sqrt(3) / 2.0
            pts.append([px, py])
            if len(pts) >= N + 10:
                break
        if len(pts) >= N + 10:
            break
    pts = np.array(pts[:N])
    
    angle = rng.uniform(-0.3, 0.3)
    ca, sa = np.cos(angle), np.sin(angle)
    rot = pts @ np.array([[ca, -sa], [sa, ca]])
    
    rot -= rot.min(axis=0)
    rng_val = rot.max(axis=0) - rot.min(axis=0)
    rng_val = np.maximum(rng_val, 1e-10)
    rot = rot / rng_val * 0.85 + 0.075
    rot += rng.normal(0, 0.003, rot.shape)
    rot = np.clip(rot, 0.05, 0.95)
    return rot

def repair_radii(centers, radii, tol=1e-9):
    radii = radii.copy()
    for _ in range(50):
        changed = False
        for i in range(N):
            mx = min(centers[i, 0], 1.0 - centers[i, 0],
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mx + tol:
                radii[i] = max(mx - tol, 0.0)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0],
                             centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d + tol:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + tol
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    bounds_xy = [(0.01, 0.99)] * (2 * N)
    
    # Generate diverse starting configurations
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 5, 5, 5, 5], [5, 5, 6, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [4, 5, 6, 5, 6], [6, 4, 6, 5, 5],
        [5, 5, 4, 6, 6], [6, 5, 4, 6, 5], [4, 5, 5, 6, 6],
        [5, 5, 5, 6, 5], [6, 6, 4, 5, 5], [5, 5, 6, 6, 4]
    ]
    
    for pat in patterns:
        for r_est in [0.085, 0.09, 0.095, 0.10, 0.105, 0.11]:
            starts.append(generate_hex_config(pat, r_est, rng, 0.003))
    
    for _ in range(10):
        starts.append(generate_corner_config(rng))
    
    for _ in range(8):
        starts.append(generate_rotated_hex(rng))
    
    for _ in range(15):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
    
    # Phase 1: L-BFGS-B gradient ascent from multiple starts
    for c0 in starts:
        v0 = c0.flatten()
        try:
            res = minimize(lp_obj_and_grad, v0, jac=True,
                          method='L-BFGS-B', bounds=bounds_xy,
                          options={'maxiter': 1000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            _, s_opt = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Powell optimization on centers with LP objective
    if best_c is not None:
        res_p = minimize(obj_func_centers, best_c.flatten(), method='Powell',
                        bounds=bounds_xy, options={'maxiter': 2000, 'ftol': 1e-14})
        c_p = res_p.x.reshape(N, 2)
        r_p, s_p = solve_lp_radii(c_p)
        if s_p > best_sum:
            best_sum = s_p
            best_c = c_p
            best_r = r_p
    
    # Phase 3: Perturbation search with Powell
    for _ in range(15):
        c_tr = best_c + rng.normal(0, 0.004, best_c.shape)
        c_tr = np.clip(c_tr, 0.02, 0.98)
        res_p2 = minimize(obj_func_centers, c_tr.flatten(), method='Powell',
                         bounds=bounds_xy, options={'maxiter': 1000, 'ftol': 1e-14})
        c2 = res_p2.x.reshape(N, 2)
        r2, s2 = solve_lp_radii(c2)
        if s2 > best_sum:
            best_sum = s2
            best_c = c2
            best_r = r2
    
    # Phase 4: Simulated Annealing on centers
    c_curr = best_c.copy()
    s_curr = best_sum
    r_curr, _ = solve_lp_radii(c_curr)
    T = 0.006
    for step in range(2000):
        T *= 0.9995
        c_new = c_curr + rng.normal(0, T, c_curr.shape)
        c_new = np.clip(c_new, 0.02, 0.98)
        r_new, s_new = solve_lp_radii(c_new)
        
        if s_new > s_curr:
            c_curr = c_new.copy()
            s_curr = s_new
            r_curr = r_new.copy()
            if s_curr > best_sum:
                best_sum = s_curr
                best_c = c_curr.copy()
                best_r = r_curr.copy()
        else:
            delta = s_new - s_curr
            if delta > 0 or rng.random() < np.exp(delta / max(T * 3.0, 1e-10)):
                c_curr = c_new.copy()
                s_curr = s_new
                r_curr = r_new.copy()
    
    # Phase 5: SLSQP joint polish
    def obj_joint(v):
        return -np.sum(v[2 * N:])
    
    def cons_joint(v):
        c = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
        con_list = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
        idx_i, idx_j = np.triu_indices(N, 1)
        dx = c[idx_i, 0] - c[idx_j, 0]
        dy = c[idx_i, 1] - c[idx_j, 1]
        dr = r[idx_i] + r[idx_j]
        con_list.append(np.sqrt(dx**2 + dy**2) - dr)
        return np.concatenate(con_list)
    
    r_lp, _ = solve_lp_radii(best_c)
    v0_joint = np.concatenate([best_c.flatten(), r_lp])
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res_j = minimize(obj_joint, v0_joint, method='SLSQP', bounds=bounds_joint,
                         constraints={'type': 'ineq', 'fun': cons_joint},
                         options={'maxiter': 6000, 'ftol': 1e-14})
        if np.min(cons_joint(res_j.x)) >= -1e-9:
            s = np.sum(res_j.x[2 * N:])
            if s > best_sum:
                best_sum = s
                best_c = res_j.x[:2 * N].reshape(N, 2).copy()
                best_r = res_j.x[2 * N:].copy()
    except Exception:
        pass
    
    # Phase 6: Additional perturbation refinements from SA best
    c_best_sa = best_c.copy()
    for _ in range(8):
        c_tr2 = c_best_sa + rng.normal(0, 0.002, c_best_sa.shape)
        c_tr2 = np.clip(c_tr2, 0.02, 0.98)
        try:
            res_p3 = minimize(obj_func_centers, c_tr2.flatten(), method='Powell',
                            bounds=bounds_xy, options={'maxiter': 800, 'ftol': 1e-14})
            c3 = res_p3.x.reshape(N, 2)
            r3, s3 = solve_lp_radii(c3)
            if s3 > best_sum:
                best_sum = s3
                best_c = c3
                best_r = r3
        except Exception:
            pass
    
    # Phase 7: One more SLSQP polish on best configuration
    r_final_lp, _ = solve_lp_radii(best_c)
    v0_final = np.concatenate([best_c.flatten(), r_final_lp])
    try:
        res_f = minimize(obj_joint, v0_final, method='SLSQP', bounds=bounds_joint,
                         constraints={'type': 'ineq', 'fun': cons_joint},
                         options={'maxiter': 4000, 'ftol': 1e-14})
        if np.min(cons_joint(res_f.x)) >= -1e-9:
            s_final = np.sum(res_f.x[2 * N:])
            if s_final > best_sum:
                best_sum = s_final
                best_c = res_f.x[:2 * N].reshape(N, 2).copy()
                best_r = res_f.x[2 * N:].copy()
    except Exception:
        pass
    
    # Phase 8: Strict numerical repair
    centers = best_c.copy()
    radii = best_r.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0],
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum

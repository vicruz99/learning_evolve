# sol_000145 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000121 (state 8b7edc5c) state=68ffbe08 sum of radii=2.558434 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_INDICES = np.triu_indices(N, k=1)
NUM_PAIRS = N * (N - 1) // 2
A_LP = None

def init_lp_matrix():
    global A_LP
    M = NUM_PAIRS + 4 * N
    A = np.zeros((M, N))
    idx = 0
    for i, j in zip(TRIU_INDICES[0], TRIU_INDICES[1]):
        A[idx, i] = 1.0
        A[idx, j] = 1.0
        idx += 1
    for i in range(N):
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
    A_LP = A

init_lp_matrix()

def solve_lp(centers):
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    idx = 0
    for i, j in zip(TRIU_INDICES[0], TRIU_INDICES[1]):
        b[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=(0, None), method='highs')
    return res.x if res.success else np.full(N, 0.01)

def compute_fast_sum(centers):
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    d = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(d, np.inf)
    p = 0.5 * np.min(d, axis=1)
    return np.sum(np.minimum(b, p))

def obj_powell(x):
    return -compute_fast_sum(x.reshape(N, 2))

def generate_inits(rng):
    inits = []
    # Random starts
    for _ in range(15):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
    # Hexagonal lattices
    for r0 in [0.09, 0.095, 0.10, 0.105, 0.11]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N:
            x = r0 if row % 2 == 0 else 2 * r0
            while x + r0 <= 1.0 and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        c = np.array(pts[:N])
        c += rng.normal(0, 0.003, c.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)
    # Perturbed grids
    for _ in range(5):
        gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
        c = np.column_stack((gx.flatten(), gy.flatten()))
        c = np.vstack([c, [0.5, 0.5]])[:N]
        c += rng.normal(0, 0.02, c.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)
    return inits

def repair(centers, radii):
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-11:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-11
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
    radii = np.maximum(radii, 0.0)
    return radii

def joint_obj(v):
    return -np.sum(v[2*N:])

def joint_cons(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = []
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    dx = c[:, 0][:, None] - c[:, 0][None, :]
    dy = c[:, 1][:, None] - c[:, 1][None, :]
    dr = r[:, None] + r[None, :]
    d2 = dx**2 + dy**2
    cons.append(d2[TRIU_INDICES] - dr[TRIU_INDICES]**2)
    return np.concatenate(cons)

def run_packing():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_sum = -np.inf
    best_radii = None
    
    inits = generate_inits(rng)
    bounds_c = [(0.01, 0.99)] * (2 * N)
    
    # Phase 1: Powell optimization on diverse starts
    for c0 in inits:
        res = minimize(obj_powell, c0.flatten(), method='Powell', bounds=bounds_c,
                       options={'maxiter': 3000, 'ftol': 1e-13, 'xtol': 1e-13})
        c_opt = res.x.reshape(N, 2)
        r_opt = solve_lp(c_opt)
        s = np.sum(r_opt)
        if s > best_sum:
            best_sum = s
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
    # Phase 2: Basin hopping to escape local minima
    current_centers = best_centers.copy()
    
    for step in range(60):
        noise = 0.008 * (0.95 ** step)
        c_trial = current_centers + rng.normal(0, noise, current_centers.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        res = minimize(obj_powell, c_trial.flatten(), method='Powell', bounds=bounds_c,
                       options={'maxiter': 1500, 'ftol': 1e-13, 'xtol': 1e-13})
        c_opt = res.x.reshape(N, 2)
        r_opt = solve_lp(c_opt)
        s = np.sum(r_opt)
        
        if s > best_sum + 1e-7:
            best_sum = s
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            current_centers = c_opt.copy()
        elif rng.random() < 0.15:
            current_centers = c_opt.copy()
            
    # Phase 3: Joint SLSQP refinement
    v0 = np.concatenate([best_centers.flatten(), best_radii])
    bounds_j = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_j = {'type': 'ineq', 'fun': joint_cons}
    
    try:
        res_sl = minimize(joint_obj, v0, method='SLSQP', bounds=bounds_j,
                          constraints=cons_j, options={'maxiter': 8000, 'ftol': 1e-13})
        if -res_sl.fun > best_sum - 1e-4:
            c_sl = res_sl.x[:2*N].reshape(N, 2)
            r_lp = solve_lp(c_sl)
            if np.sum(r_lp) >= -res_sl.fun:
                best_centers = c_sl
                best_radii = r_lp
                best_sum = np.sum(r_lp)
            else:
                best_centers = c_sl
                best_radii = res_sl.x[2*N:]
                best_sum = -res_sl.fun
    except Exception:
        pass
        
    # Phase 4: Strict numerical repair
    best_radii = repair(best_centers, best_radii)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum

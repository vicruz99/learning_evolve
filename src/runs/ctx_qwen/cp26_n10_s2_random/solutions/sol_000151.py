# sol_000151 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000141 (state d8f6c168) state=e707f57d sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)
NUM_PAIRS = len(I)

# Precompute LP constraint matrix structure globally
A_ub_lp = np.zeros((NUM_PAIRS + 4*N, N))
idx = 0
for i, j in zip(I, J):
    A_ub_lp[idx, i] = 1.0
    A_ub_lp[idx, j] = 1.0
    idx += 1
for i in range(N):
    for _ in range(4):
        A_ub_lp[idx, i] = 1.0
        idx += 1

def get_lp_radii(centers):
    """Solves LP to find optimal radii for fixed centers."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b = np.zeros(NUM_PAIRS + 4*N)
    idx = 0
    for i, j in zip(I, J):
        b[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(N), A_ub=A_ub_lp, b_ub=b, bounds=[(0, u) for u in ub], method='highs')
    return res.x if res.success else ub * 0.1

def compute_obj(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def compute_cons(v):
    """Computes all boundary and non-overlap constraints."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    dx = c[I, 0] - c[J, 0]
    dy = c[I, 1] - c[J, 1]
    con.append(np.sqrt(dx**2 + dy**2) - (r[I] + r[J]))
    return np.concatenate(con)

def generate_starts(n, rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5]
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
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.02, 0.98)
            starts.append(c)
            
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_v = None
    best_sum = -np.inf
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_cons}
    
    starts = generate_starts(N, rng)
    
    # Phase 1: Multi-start SLSQP optimization
    for c0 in starts:
        r0 = get_lp_radii(c0)
        v0 = np.concatenate([c0.flatten(), r0])
        v0 += rng.normal(0, 1e-6, v0.shape)
        
        try:
            res = minimize(compute_obj, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_sum:
                c_vals = compute_cons(res.x)
                if np.min(c_vals) > -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Fallback initialization if optimization fails
    if best_v is None:
        v0 = np.zeros(3*N)
        v0[0::3] = 0.5; v0[1::3] = 0.5; v0[2::3] = 0.01
        res = minimize(compute_obj, v0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 2000})
        best_v = res.x
        
    # Phase 2: Perturbation search to escape local minima
    for _ in range(40):
        v_trial = best_v + rng.normal(0, 0.0004, best_v.shape)
        v_trial[:2*N] = np.clip(v_trial[:2*N], 0.0, 1.0)
        v_trial[2*N:] = np.clip(v_trial[2*N:], 0.0, 0.5)
        try:
            res = minimize(compute_obj, v_trial, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-13})
            if -res.fun > best_sum:
                c_vals = compute_cons(res.x)
                if np.min(c_vals) > -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass

    # Phase 3: Extract centers and solve exact LP for radii
    best_centers = best_v[:2*N].reshape(N, 2)
    best_radii = get_lp_radii(best_centers)
    best_sum = np.sum(best_radii)
    
    # Phase 4: Strict numerical repair to guarantee validator tolerance compliance
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                if d < best_radii[i] + best_radii[j] - 1e-12:
                    shrink = (best_radii[i] + best_radii[j] - d) / 2.0 + 1e-9
                    best_radii[i] -= shrink
                    best_radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(best_centers[i,0], 1.0-best_centers[i,0], best_centers[i,1], 1.0-best_centers[i,1])
            if best_radii[i] > mr + 1e-12:
                best_radii[i] = mr
                changed = True
        if not changed:
            break
            
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(np.sum(best_radii))

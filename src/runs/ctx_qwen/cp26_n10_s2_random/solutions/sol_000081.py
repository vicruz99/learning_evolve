# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000058 (state f7fedeb3) state=17bb691d sum of radii=1.770102 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
# Precompute pairwise indices for efficient constraint evaluation
IDX_I, IDX_J = np.triu_indices(N_CIRCLES, k=1)

def compute_obj(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N_CIRCLES:])

def compute_cons(v):
    """Inequality constraints: boundary and non-overlap (must be >= 0)."""
    c = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r = v[2*N_CIRCLES:]
    
    # Boundary constraints
    out = [
        c[:, 0] - r,          # x >= r
        1.0 - c[:, 0] - r,    # x + r <= 1
        c[:, 1] - r,          # y >= r
        1.0 - c[:, 1] - r     # y + r <= 1
    ]
    
    # Pairwise non-overlap constraints
    xi, xj = c[IDX_I, 0], c[IDX_J, 0]
    yi, yj = c[IDX_I, 1], c[IDX_J, 1]
    ri, rj = r[IDX_I], r[IDX_J]
    out.append(np.hypot(xi - xj, yi - yj) - (ri + rj))
    
    return np.concatenate(out)

def get_hex_init(seed):
    """Generates a perturbed hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    c = np.zeros((N_CIRCLES, 2))
    r_est = 0.1
    y = r_est
    row = 0
    idx = 0
    while idx < N_CIRCLES:
        x = r_est + (row % 2) * r_est
        while x <= 1.0 - r_est and idx < N_CIRCLES:
            c[idx] = [x, y]
            x += 2 * r_est
            idx += 1
        y += np.sqrt(3) * r_est
        row += 1
    # Add controlled noise to break symmetry
    c += rng.normal(0, 0.005, c.shape)
    return np.clip(c, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_v = None
    best_sum = -1.0
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': compute_cons}
    
    # Phase 1: Multi-start with path-following growth strategy
    for s in range(25):
        c0 = get_hex_init(s)
        r0 = np.full(N_CIRCLES, 0.035)  # Start feasible
        v0 = np.concatenate([c0.flatten(), r0])
        
        # Gradually grow radii while optimizing centers
        for _ in range(10):
            v0[2*N_CIRCLES:] *= 1.02  # Grow by 2%
            try:
                res = minimize(compute_obj, v0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
                # Check feasibility and update best
                if -res.fun > best_sum and np.all(compute_cons(res.x) >= -1e-9):
                    best_sum = -res.fun
                    best_v = res.x.copy()
                v0 = res.x.copy()
            except Exception:
                pass
                
    # Fallback if optimization failed
    if best_v is None:
        c_fb = get_hex_init(0)
        best_v = np.concatenate([c_fb.flatten(), np.full(N_CIRCLES, 0.08)])
        
    # Phase 2: Intensive local polishing around the best solution
    for _ in range(15):
        v_pert = best_v.copy()
        v_pert[:2*N_CIRCLES] += rng.normal(0, 0.0005, 2*N_CIRCLES)
        try:
            res = minimize(compute_obj, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-13})
            if -res.fun > best_sum and np.all(compute_cons(res.x) >= -1e-9):
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass

    centers = best_v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_v[2*N_CIRCLES:].copy()
    
    # Phase 3: Exact LP refinement for optimal radii given final centers
    centers = np.clip(centers, 0.0, 1.0)
    n_pairs = N_CIRCLES * (N_CIRCLES - 1) // 2
    n_bnd = 4 * N_CIRCLES
    A_ub = np.zeros((n_pairs + n_bnd, N_CIRCLES))
    k = 0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            k += 1
    for i in range(N_CIRCLES):
        for _ in range(4):
            A_ub[k, i] = 1.0
            k += 1
            
    c_obj = -np.ones(N_CIRCLES)
    b_ub = np.zeros(n_pairs + n_bnd)
    k = 0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            b_ub[k] = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            k += 1
        x, y = centers[i]
        b_ub[k] = x; k += 1
        b_ub[k] = 1.0 - x; k += 1
        b_ub[k] = y; k += 1
        b_ub[k] = 1.0 - y; k += 1
        
    res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    if res_lp.success:
        radii = res_lp.x
        
    # Phase 4: Strict deterministic repair for validation tolerance
    for _ in range(5):
        changed = False
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    sh = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] -= sh
                    radii[j] -= sh
                    changed = True
        for i in range(N_CIRCLES):
            mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mx + 1e-12:
                radii[i] = mx
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))

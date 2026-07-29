# sol_000148 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000121 (state 8b7edc5c) state=1f6bc330 sum of radii=0.530601 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_INDICES = np.triu_indices(N, k=1)

def compute_slsqp_obj(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def compute_slsqp_cons(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints
    c = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    c.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(c)

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def generate_scaled_hex(r_target, rng):
    """Generates a hexagonal lattice configuration scaled to fit r_target."""
    pts = []
    y = 0.0
    row = 0
    while len(pts) < N:
        x_start = (row % 2) * r_target
        x = x_start
        while len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_target
        y += np.sqrt(3) * r_target
        row += 1
        
    pts = np.array(pts[:N])
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    span = max_pt - min_pt
    
    # Scale to fit tightly within [r, 1-r]
    scale = min((1.0 - 2.0 * r_target) / span[0], (1.0 - 2.0 * r_target) / span[1])
    pts = (pts - min_pt) * scale + r_target
    
    # Add small perturbation to break symmetry
    pts += rng.normal(0, 0.002, pts.shape)
    return np.clip(pts, 0.01, 0.99)

def solve_lp_radii(centers):
    """Solves LP to find optimal radii for fixed centers."""
    n = centers.shape[0]
    n_pairs = n * (n - 1) // 2
    n_bnd = 4 * n
    
    A_ub = np.zeros((n_pairs + n_bnd, n))
    b_ub = np.zeros(n_pairs + n_bnd)
    
    idx = 0
    # Pairwise constraints
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 0.0)
    
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        for k in range(4):
            A_ub[idx, i] = 1.0
            b_ub[idx] = [x, 1.0 - x, y, 1.0 - y][k]
            idx += 1
            
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    if res.success:
        return res.x
    return np.full(n, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize sum of radii."""
    rng = np.random.default_rng(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_slsqp_cons}
    
    best_v = None
    best_sum = -1e9
    
    # Phase 1: Multi-start SLSQP from scaled hexagonal lattices
    # Targeting radii ~0.105 places us near the known optimal packing density
    for r_init in [0.095, 0.100, 0.105, 0.108]:
        for _ in range(4):
            c = generate_scaled_hex(r_init, rng)
            v0 = np.zeros(3 * N)
            v0[0::3] = c[:, 0]
            v0[1::3] = c[:, 1]
            # Start with slightly smaller radii to guarantee strict feasibility
            v0[2::3] = r_init * 0.95
            
            try:
                res = minimize(compute_slsqp_obj, v0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 25000, 'ftol': 1e-14})
                s_val = -res.fun
                if s_val > best_sum:
                    c_vals = compute_slsqp_cons(res.x)
                    if np.min(c_vals) >= -1e-7:
                        best_sum = s_val
                        best_v = res.x.copy()
            except Exception:
                pass
                
    if best_v is None:
        c = generate_scaled_hex(0.09, rng)
        best_v = np.zeros(3 * N)
        best_v[0::3] = c[:, 0]
        best_v[1::3] = c[:, 1]
        best_v[2::3] = 0.09
        
    # Phase 2: Hill-climbing on centers with exact LP radius optimization
    # Decoupling centers and radii allows more aggressive exploration
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = solve_lp_radii(centers)
    current_sum = np.sum(radii)
    
    step = 0.005
    for _ in range(150):
        c_trial = centers.copy()
        c_trial += rng.normal(0, step, c_trial.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        r_trial = solve_lp_radii(c_trial)
        s_trial = np.sum(r_trial)
        
        if s_trial > current_sum + 1e-9:
            current_sum = s_trial
            centers = c_trial.copy()
            radii = r_trial.copy()
            step = min(step * 1.05, 0.015)  # Increase step on success
        else:
            step *= 0.95  # Decrease step on failure
            
    # Phase 3: Final joint SLSQP polish for micro-adjustments
    v_final = np.zeros(3 * N)
    v_final[0::3] = centers[:, 0]
    v_final[1::3] = centers[:, 1]
    v_final[2::3] = radii
    
    try:
        res = minimize(compute_slsqp_obj, v_final, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
        if -res.fun > current_sum - 1e-6:
            centers = np.column_stack((res.x[0::3], res.x[1::3]))
            radii = solve_lp_radii(centers)
    except Exception:
        pass
        
    # Phase 4: Strict numerical repair to guarantee validation tolerance compliance
    for _ in range(50):
        changed = False
        # Resolve pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-11:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-11
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Resolve boundary violations
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
                
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum

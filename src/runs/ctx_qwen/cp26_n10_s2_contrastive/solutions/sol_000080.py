# sol_000080 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000051 (state 921aef56) state=9d558f79 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, 1)

def solve_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((n*(n-1)//2, n))
    b_ub = np.zeros(n*(n-1)//2)
    
    # Vectorized pairwise distances
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists = np.sqrt(dx*dx + dy*dy)
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0-x, y, 1.0-y)
        bounds_r.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success and np.all(res.x >= -1e-7):
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + len(I_IDX))
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - r[I_IDX] - r[J_IDX]
    return c

def make_hex_init(seed, spacing=0.18, margin=0.05):
    """Generate perturbed hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin
    while idx < N and y < 1.0 - margin:
        x = margin + (row % 2) * spacing / 2.0
        while x < 1.0 - margin and idx < N:
            centers[idx] = [x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01)]
            idx += 1
            x += spacing
        y += spacing * np.sqrt(3) / 2.0
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin, 1-margin, 2)
        idx += 1
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Phase 1: Generate diverse initial configurations
    starts = []
    # Hexagonal patterns with varying spacing
    for s in range(15):
        spacing = 0.155 + s * 0.005
        starts.append(make_hex_init(s, spacing=spacing))
        
    # Random uniform placements
    for s in range(15):
        rng = np.random.RandomState(s*7+1)
        starts.append(rng.uniform(0.12, 0.88, (N, 2)))
        
    # Phase 2: SLSQP + LP refinement loop
    for c_init in starts:
        r_init, _ = solve_lp(c_init)
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = np.maximum(r_init, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                cc = np.column_stack((cx, cy))
                rr, ss = solve_lp(cc)
                if ss > best_sum:
                    best_sum = ss
                    best_c = cc
                    best_r = rr
        except Exception:
            continue
            
    # Phase 3: Local perturbation refinement around best solution
    if best_c is not None:
        rng = np.random.RandomState(42)
        for _ in range(25):
            # Perturb centers
            c_pert = best_c + rng.normal(0, 0.004, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert, _ = solve_lp(c_pert)
            
            x0 = np.zeros(3*N)
            x0[0::3] = c_pert[:, 0]
            x0[1::3] = c_pert[:, 1]
            x0[2::3] = np.maximum(r_pert, 1e-5)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
                if res.success:
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    cc = np.column_stack((cx, cy))
                    rr, ss = solve_lp(cc)
                    if ss > best_sum:
                        best_sum = ss
                        best_c = cc
                        best_r = rr
            except Exception:
                continue

    # Fallback safety net
    if best_c is None:
        best_c = make_hex_init(0)
        best_r, best_sum = solve_lp(best_c)
        
    # Phase 4: Strict numerical validation and adjustment
    radii = best_r.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        x, y = best_c[i]
        max_r = min(x, 1-x, y, 1-y)
        if radii[i] > max_r - 1e-9:
            radii[i] = max(0.0, max_r - 1e-9)
            
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
                if d < radii[i] + radii[j] - 1e-9:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc/2.0
                    radii[j] -= exc/2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_c, radii, float(np.sum(radii))

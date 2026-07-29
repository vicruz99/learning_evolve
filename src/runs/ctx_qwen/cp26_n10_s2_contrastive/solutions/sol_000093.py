# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000054 (state 65bdd474) state=ace78e4c sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def compute_lp_radii(centers):
    """Solve LP to find optimal radii for fixed centers."""
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists = np.hypot(dx, dy)
    
    A_ub = np.zeros((NUM_PAIRS, N))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(N):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(N), 0.0

def objective(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[2*N:])

def constraints(params):
    """Inequality constraints: boundary and non-overlap (must be >= 0)."""
    c = params[:2*N].reshape(N, 2)
    r = params[2*N:]
    
    b = np.empty(4*N)
    b[0:N] = c[:, 0] - r
    b[N:2*N] = 1.0 - c[:, 0] - r
    b[2*N:3*N] = c[:, 1] - r
    b[3*N:4*N] = 1.0 - c[:, 1] - r
    
    diff = c[:, None, :] - c[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    overlaps = dists[I_IDX, J_IDX] - (r[I_IDX] + r[J_IDX])
    
    return np.concatenate([b, overlaps])

def run_packing():
    bounds_opt = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Generate diverse structured initializations
    inits = []
    for seed in range(40):
        rng = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        
        # Hexagonal lattice with randomized spacing, margin, and symmetry breaking
        margin = rng.uniform(0.02, 0.06)
        sp = rng.uniform(0.16, 0.20)
        y = margin
        row = 0
        while idx < N and y < 1.0 - margin:
            x = margin + (row % 2) * sp / 2.0
            while x < 1.0 - margin and idx < N:
                c[idx] = [x + rng.normal(0, 0.005), y + rng.normal(0, 0.005)]
                x += sp
                idx += 1
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = rng.uniform(margin, 1.0 - margin, 2)
            idx += 1
        inits.append(c)

    # Phase 1: Joint optimization with SLSQP
    for c0 in inits:
        r0, _ = compute_lp_radii(c0)
        r0 = np.maximum(r0, 1e-5) * 0.95  # Shrink slightly to ensure strict feasibility
        x0 = np.concatenate([c0.flatten(), r0])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_dict,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            
            co = res.x[:2*N].reshape(N, 2)
            ro, so = compute_lp_radii(co)
            if so > best_sum:
                best_sum = so
                best_c = co.copy()
                best_r = ro.copy()
        except Exception:
            pass

    # Phase 2: Direct local search on centers using exact LP evaluation
    if best_c is not None:
        step = 0.006
        for trial in range(900):
            rng = np.random.RandomState(trial * 19 + 3)
            # Occasional large jumps to escape basins
            if rng.rand() < 0.05:
                step_cur = rng.uniform(0.01, 0.03)
            else:
                step_cur = step
                
            c_pert = best_c + rng.randn(N, 2) * step_cur
            c_pert = np.clip(c_pert, 0.005, 0.995)
            rp, sp = compute_lp_radii(c_pert)
            
            if sp > best_sum:
                best_sum = sp
                best_c = c_pert.copy()
                best_r = rp.copy()
                step *= 1.08  # Expand step on improvement
                step = min(step, 0.025)
            else:
                step *= 0.94  # Contract step on failure
                step = max(step, 1e-4)

    # Phase 3: Strict post-processing to guarantee validity
    if best_c is not None:
        # Enforce boundary constraints strictly
        for i in range(N):
            mx = min(best_c[i,0], 1.0 - best_c[i,0], best_c[i,1], 1.0 - best_c[i,1])
            if best_r[i] > mx:
                best_r[i] = max(0.0, mx - 1e-9)
                
        # Iteratively resolve remaining numerical overlaps
        for _ in range(100):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(best_c[i,0] - best_c[j,0], best_c[i,1] - best_c[j,1])
                    if d < best_r[i] + best_r[j] - 1e-10:
                        exc = best_r[i] + best_r[j] - d
                        best_r[i] -= exc / 2.0
                        best_r[j] -= exc / 2.0
                        changed = True
            if not changed:
                break
        best_sum = float(np.sum(best_r))
        
    # Fallback safety net
    if best_c is None:
        best_c = np.random.uniform(0.15, 0.85, (N, 2))
        best_r, best_sum = compute_lp_radii(best_c)
        
    return best_c, best_r, float(best_sum)

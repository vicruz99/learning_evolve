# sol_000270 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000250 (state 61d3a642) state=12cd5a7b sum of radii=2.608186 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
IDX_I, IDX_J = np.triu_indices(N, k=1)
N_PAIRS = len(IDX_I)

# Precompute LP constraint matrix structure
A_UB_LP = np.zeros((N_PAIRS, N))
A_UB_LP[np.arange(N_PAIRS), IDX_I] = 1.0
A_UB_LP[np.arange(N_PAIRS), IDX_J] = 1.0

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    lims = np.minimum(np.minimum(centers[:,0], 1.0-centers[:,0]),
                      np.minimum(centers[:,1], 1.0-centers[:,1]))
    bounds = [(0.0, max(l, 1e-9)) for l in lims]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[IDX_I, IDX_J]
    try:
        res = linprog(-np.ones(N), A_ub=A_UB_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(N, 1e-6), 0.0

def obj_joint(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Inequality constraints >= 0 for valid joint packing."""
    cx = v[:N]
    cy = v[N:2*N]
    r = v[2*N:]
    # Boundary constraints
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    # Pairwise non-overlap (squared)
    dx = cx[IDX_I] - cx[IDX_J]
    dy = cy[IDX_I] - cy[IDX_J]
    dr = r[IDX_I] + r[IDX_J]
    return np.concatenate([c, dx**2 + dy**2 - dr**2])

def make_hex_config(pat, r0):
    """Generates a hexagonal lattice configuration with specified row distribution."""
    pts = []
    y = r0
    for i, cnt in enumerate(pat):
        shift = r0 * 0.5 if i % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= N: break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
    pts = np.array(pts[:N])
    # Normalize to fit comfortably inside [0.1, 0.9]
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn + 1e-9
    pts = (pts - mn) / span * 0.7 + 0.15
    return np.clip(pts, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0)] * (2 * N) + [(1e-5, 0.5)] * N
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Diverse structural patterns known to be competitive for N=26
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4], 
        [6,6,5,5,4], [5,6,4,6,5], [7,5,5,5,4], [4,5,7,5,5]
    ]
    
    inits = []
    for pat in patterns:
        if sum(pat) != N: continue
        for r0 in [0.08, 0.09, 0.10, 0.11]:
            inits.append(make_hex_config(pat, r0))
            
    # Add randomized starts to escape lattice biases
    for _ in range(15):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Phase 1: Multi-start SLSQP to find high-quality basins
    for cfg in inits:
        # Ensure initial feasibility for radii
        lims = np.minimum(np.minimum(cfg[:,0], 1.0-cfg[:,0]), np.minimum(cfg[:,1], 1.0-cfg[:,1]))
        diff = cfg[:, None, :] - cfg[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists, axis=1) / 2.0
        r_init = np.minimum(lims, min_d) * 0.95
        
        v0 = np.concatenate([cfg[:,0], cfg[:,1], r_init])
        
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_opt,
                           constraints={'type': 'ineq', 'fun': cons_joint},
                           options={'maxiter': 3000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                cx = res.x[:N]
                cy = res.x[N:2*N]
                c_opt = np.column_stack((cx, cy))
                r_lp, s_lp = solve_lp(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_c = c_opt.copy()
                    best_r = r_lp.copy()
        except Exception:
            pass

    # Fallback if optimizers fail
    if best_c is None:
        best_c = inits[0]
        best_r, best_sum = solve_lp(best_c)
        
    # Phase 2: LP-guided Hill Climbing on Centers
    # Perturb centers and accept moves that increase the LP-optimal sum of radii.
    # This is highly efficient for this specific objective.
    curr_c = best_c.copy()
    curr_r = best_r.copy()
    curr_s = best_sum
    step_size = 0.014
    
    for it in range(900):
        i = rng.integers(N)
        old = curr_c[i].copy()
        
        # Random perturbation
        curr_c[i] += rng.uniform(-step_size, step_size, 2)
        curr_c[i] = np.clip(curr_c[i], 0.01, 0.99)
        
        r_try, s_try = solve_lp(curr_c)
        if s_try > curr_s + 1e-8:
            curr_s = s_try
            curr_r = r_try.copy()
        else:
            curr_c[i] = old # Revert if no improvement
            
        step_size *= 0.9985 # Gradually refine step size
        
    best_c = curr_c
    best_r = curr_r
    best_sum = curr_s
    
    # Phase 3: Final Joint SLSQP Polish
    # Allows slight structural deformation to squeeze out extra radius
    v0_pol = np.concatenate([best_c[:,0], best_c[:,1], best_r * 0.99])
    try:
        res_p = minimize(obj_joint, v0_pol, method='SLSQP', bounds=bounds_opt,
                         constraints={'type': 'ineq', 'fun': cons_joint},
                         options={'maxiter': 4000, 'ftol': 1e-13})
        if np.isfinite(res_p.fun):
            c_p = np.column_stack((res_p.x[:N], res_p.x[N:2*N]))
            r_p, s_p = solve_lp(c_p)
            if s_p > best_sum:
                best_c = c_p
                best_r = r_p
                best_sum = s_p
    except Exception:
        pass
        
    # Phase 4: Strict Numerical Safety Scaling
    # Guarantee validity within 1e-12 tolerance for the validator
    scale = 1.0
    for i in range(N):
        x, y, ri = best_c[i,0], best_c[i,1], best_r[i]
        if ri > 1e-12:
            scale = min(scale, x/ri, (1.0-x)/ri, y/ri, (1.0-y)/ri)
            
    diff = best_c[:, None, :] - best_c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_sum = best_r[:, None] + best_r[None, :]
    scale = min(scale, np.min(dists[IDX_I, IDX_J] / np.maximum(r_sum[IDX_I, IDX_J], 1e-12)))
    
    best_r *= scale * 0.9999995
    final_sum = float(np.sum(best_r))
    
    return best_c, best_r, final_sum

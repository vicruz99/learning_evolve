# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000094 (state 4cf54399) state=0b3a29be sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists = np.hypot(dx, dy)
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def objective(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (must be >= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def make_valid(centers, radii):
    """Project configuration to strictly satisfy boundary and overlap constraints."""
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-10)
            
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-12:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    return centers, np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    rng = np.random.default_rng(42)
    inits = []
    
    # Generate diverse hexagonal lattices with varying parameters
    for seed in range(50):
        r_gen = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        row = 0
        sp = 0.16 + r_gen.uniform(-0.03, 0.03)
        margin = 0.03 + r_gen.uniform(0, 0.03)
        y = margin
        while idx < N and y < 1.0 - margin:
            x = margin + (row % 2) * sp / 2.0
            while x < 1.0 - margin and idx < N:
                c[idx] = [x, y]
                x += sp
                idx += 1
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(margin, 1.0 - margin, 2)
            idx += 1
        c += r_gen.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)

    # Add corner-focused configurations to exploit boundary slack
    for seed in range(20):
        r_gen = np.random.RandomState(seed + 100)
        c = np.zeros((N, 2))
        idx = 0
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        for cc in corners:
            if idx < N:
                c[idx] = cc
                idx += 1
        sp = 0.16 + r_gen.uniform(-0.02, 0.02)
        margin = 0.15
        y = margin
        row = 0
        while idx < N and y < 1.0 - margin:
            x = margin + (row % 2) * sp / 2.0
            while x < 1.0 - margin and idx < N:
                c[idx] = [x, y]
                x += sp
                idx += 1
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(0.15, 0.85, 2)
            idx += 1
        c += r_gen.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)

    def try_optimize(c_init):
        nonlocal best_sum, best_c, best_r
        r_init = solve_lp_radii(c_init)
        r_init = np.maximum(r_init * 0.98, 1e-4)
        
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_dict, options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            co = np.column_stack((cx, cy))
            ro = solve_lp_radii(co)
            s = np.sum(ro)
            
            if s > best_sum:
                best_sum = s
                best_c = co.copy()
                best_r = ro.copy()
        except Exception:
            pass

    # Phase 1: Multi-start SLSQP optimization
    for c0 in inits:
        try_optimize(c0)
        
    # Phase 2: Adaptive Basin Hopping with LP evaluation & SLSQP polishing
    if best_c is not None:
        for step in range(200):
            noise_scale = 0.025 * np.exp(-step / 60.0) + 0.001
            c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > best_sum:
                best_c = c_pert.copy()
                best_r = r_pert.copy()
                best_sum = s_pert
                
                # Local polishing
                x0 = np.zeros(3*N)
                x0[0::3] = best_c[:, 0]
                x0[1::3] = best_c[:, 1]
                x0[2::3] = np.maximum(best_r * 0.99, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_dict, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    co = np.column_stack((cx, cy))
                    ro = solve_lp_radii(co)
                    s = np.sum(ro)
                    if s > best_sum:
                        best_c = co
                        best_r = ro
                        best_sum = s
                except Exception:
                    pass
                    
    # Fallback safety net
    if best_c is None:
        best_c = inits[0]
        best_r = solve_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
    # Strict post-processing to guarantee validity within numerical tolerance
    best_c, best_r = make_valid(best_c, best_r)
    return best_c, best_r, float(np.sum(best_r))

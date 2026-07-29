# sol_000053 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000041 (state 046a36a4) state=10ec8263 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary and pairwise non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + len(I_IDX))
    c[0:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - r[I_IDX] - r[J_IDX]
    return c

def get_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = np.zeros((n*(n-1)//2, n))
    b_ub = np.zeros(n*(n-1)//2)
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = d
            k += 1
            
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0-x, y, 1.0-y)
        bounds_r.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def validate_and_fix(centers, radii):
    """Ensure strict feasibility within numerical tolerance."""
    n = centers.shape[0]
    radii = radii.copy()
    
    for i in range(n):
        mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-9)
            
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d + 1e-10
                    radii[i] -= exc/2
                    radii[j] -= exc/2
                    changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return centers, radii

def run_packing():
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Multi-start SLSQP + LP refinement
    for seed in range(60):
        rng = np.random.RandomState(seed)
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.full(N, 0.04)
        
        if seed < 30:
            # Hexagonal lattice initialization
            row = 0
            idx = 0
            y = 0.12
            while idx < N and y < 0.9:
                x = 0.12 + (row % 2) * 0.08
                col = 0
                while x < 0.9 and idx < N:
                    cx[idx] = x + rng.uniform(-0.01, 0.01)
                    cy[idx] = y + rng.uniform(-0.01, 0.01)
                    idx += 1
                    x += 0.16
                y += 0.14
                row += 1
            while idx < N:
                cx[idx] = rng.uniform(0.1, 0.9)
                cy[idx] = rng.uniform(0.1, 0.9)
                idx += 1
        else:
            # Random initialization
            cx = rng.uniform(0.15, 0.85, N)
            cy = rng.uniform(0.15, 0.85, N)
            
        cx = np.clip(cx, 0.02, 0.98)
        cy = np.clip(cy, 0.02, 0.98)
        x0 = np.column_stack([cx, cy, r]).flatten()
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                cx_opt = res.x[0::3].copy()
                cy_opt = res.x[1::3].copy()
                centers_opt = np.column_stack([cx_opt, cy_opt])
                
                # Extract optimal radii for these centers via LP
                lp_r, lp_sum = get_lp_radii(centers_opt)
                if lp_sum > best_sum:
                    c_fix, r_fix = validate_and_fix(centers_opt, lp_r)
                    curr_sum = np.sum(r_fix)
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = c_fix.copy()
                        best_radii = r_fix.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation search around best solution
    if best_centers is not None:
        rng = np.random.RandomState(42)
        for _ in range(20):
            pert_c = best_centers + rng.randn(N, 2) * 0.005
            pert_c = np.clip(pert_c, 0.02, 0.98)
            pert_r = best_radii + rng.randn(N) * 0.002
            pert_r = np.clip(pert_r, 0.001, 0.5)
            x0_p = np.column_stack([pert_c, pert_r]).flatten()
            
            try:
                res = minimize(objective, x0_p, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                if res.success:
                    cx_opt = res.x[0::3].copy()
                    cy_opt = res.x[1::3].copy()
                    centers_opt = np.column_stack([cx_opt, cy_opt])
                    
                    lp_r, lp_sum = get_lp_radii(centers_opt)
                    if lp_sum > best_sum:
                        c_fix, r_fix = validate_and_fix(centers_opt, lp_r)
                        curr_sum = np.sum(r_fix)
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_centers = c_fix.copy()
                            best_radii = r_fix.copy()
            except Exception:
                continue
                
    # Fallback safety net
    if best_centers is None:
        best_centers = np.random.uniform(0.2, 0.8, (N, 2))
        best_radii, best_sum = get_lp_radii(best_centers)
        best_centers, best_radii = validate_and_fix(best_centers, best_radii)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)

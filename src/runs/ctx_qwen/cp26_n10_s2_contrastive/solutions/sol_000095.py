# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000054 (state 65bdd474) state=87f66e5e sum of radii=1.041678 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

def compute_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.sqrt(dx**2 + dy**2)
    
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(N_PAIRS), J_IDX] = 1.0
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.zeros(n)

def objective(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[2 * N:])

def constraints(params):
    """Inequality constraints: boundary clearance and pairwise non-overlap (squared)."""
    cx = params[0::3]
    cy = params[1::3]
    r = params[2::3]
    
    c = np.empty(4 * N + N_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = dx**2 + dy**2 - (r[I_IDX] + r[J_IDX])**2
    return c

def generate_initializations():
    """Generate diverse structured and random initial center configurations."""
    inits = []
    rng = np.random.RandomState(123)
    
    # 1. Hexagonal lattices with varying spacing
    for s in np.linspace(0.155, 0.195, 10):
        c = np.zeros((N, 2))
        idx = 0
        y = s / 2
        row = 0
        while idx < N and y < 1.0 - s / 2:
            x = s / 2 + (row % 2) * s / 2
            while x < 1.0 - s / 2 and idx < N:
                c[idx] = [x, y]
                x += s
                idx += 1
            y += s * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.01, 0.99)
        inits.append(c)
        
    # 2. Grid patterns
    for step in np.linspace(0.17, 0.21, 6):
        c = np.zeros((N, 2))
        idx = 0
        y = step
        while y < 1.0 - step and idx < N:
            x = step
            while x < 1.0 - step and idx < N:
                c[idx] = [x, y]
                x += step
                idx += 1
            y += step
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        c += rng.normal(0, 0.004, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    # 3. Random feasible starts
    for _ in range(20):
        c = rng.uniform(0.05, 0.95, (N, 2))
        inits.append(c)
        
    return inits

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Uses a hybrid SLSQP + LP strategy with multiple restarts and local refinement.
    """
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    inits = generate_initializations()
    
    for c0 in inits:
        # Initialize radii via LP, scale slightly down to ensure strict feasibility for SLSQP
        r0 = compute_lp_radii(c0) * 0.98
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 18000, 'ftol': 1e-13, 'disp': False})
            
            co = res.x[:2 * N].reshape(N, 2)
            # Extract exact optimal radii for these centers via LP
            ro = compute_lp_radii(co)
            s_lp = np.sum(ro)
            
            if s_lp > best_sum:
                best_sum = s_lp
                best_c = co.copy()
                best_r = ro.copy()
                
                # Local perturbation refinement to escape shallow minima
                for trial in range(8):
                    rng_p = np.random.RandomState(trial * 31 + 7)
                    cp = best_c + rng_p.randn(N, 2) * (0.004 * (1.0 / (1 + trial)))
                    cp = np.clip(cp, 0.015, 0.985)
                    rp = compute_lp_radii(cp)
                    xp = np.zeros(3 * N)
                    xp[0::3] = cp[:, 0]
                    xp[1::3] = cp[:, 1]
                    xp[2::3] = rp * 0.99
                    
                    try:
                        res2 = minimize(objective, xp, method='SLSQP', bounds=bounds, constraints=cons_dict,
                                       options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                        c2 = res2.x[:2 * N].reshape(N, 2)
                        r2 = compute_lp_radii(c2)
                        s2 = np.sum(r2)
                        if s2 > best_sum:
                            best_sum = s2
                            best_c = c2.copy()
                            best_r = r2.copy()
                    except Exception:
                        pass
        except Exception:
            pass

    # Strict post-processing to guarantee validity within numerical tolerance
    if best_c is not None:
        margin = 1e-9
        for i in range(N):
            mx = min(best_c[i, 0], 1.0 - best_c[i, 0], best_c[i, 1], 1.0 - best_c[i, 1])
            best_r[i] = min(best_r[i], mx - margin)
            best_r[i] = max(best_r[i], 0.0)
            
        # Iteratively resolve any remaining numerical overlaps
        for _ in range(150):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
                    if d < best_r[i] + best_r[j] - 1e-10:
                        exc = best_r[i] + best_r[j] - d
                        best_r[i] -= exc / 2.0
                        best_r[j] -= exc / 2.0
                        changed = True
            if not changed:
                break
        best_sum = np.sum(best_r)
        
    # Fallback safety net (should not be reached with proper initialization)
    if best_c is None:
        best_c = np.random.uniform(0.1, 0.9, (N, 2))
        best_r = compute_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
    return best_c, best_r, float(best_sum)

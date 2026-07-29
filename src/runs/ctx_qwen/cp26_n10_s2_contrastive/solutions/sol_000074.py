# sol_000074 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000036 (state d4cf115e) state=1f0b3fc0 sum of radii=2.565215 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
N_PAIRS = len(PAIRS_I)

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    diff = centers[PAIRS_I] - centers[PAIRS_J]
    dists = np.hypot(diff[:, 0], diff[:, 1])
    
    c_obj = -np.ones(n)
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), PAIRS_I] = 1.0
    A_ub[np.arange(N_PAIRS), PAIRS_J] = 1.0
    b_ub = dists
    
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + N_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[PAIRS_I] - cx[PAIRS_J]
    dy = cy[PAIRS_I] - cy[PAIRS_J]
    c[4*N:] = np.hypot(dx, dy) - r[PAIRS_I] - r[PAIRS_J]
    return c

def make_valid(centers, radii):
    """Post-process to strictly satisfy boundary and overlap constraints."""
    centers = np.clip(centers, 0.0, 1.0)
    radii = np.maximum(radii, 0.0)
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-9)
            
    # Iteratively resolve overlaps
    for _ in range(100):
        changed = False
        for k in range(N_PAIRS):
            i, j = PAIRS_I[k], PAIRS_J[k]
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < radii[i] + radii[j] - 1e-10:
                excess = radii[i] + radii[j] - d
                radii[i] -= excess * 0.5
                radii[j] -= excess * 0.5
                changed = True
        if not changed:
            break
    return centers, radii

def generate_initializations():
    """Create diverse starting points for optimization."""
    inits = []
    rng_base = np.random.RandomState(42)
    
    # 1. Hexagonal lattices with varying spacing and offsets
    for seed in range(25):
        rng = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        row = 0
        spacing = 0.16 + rng.uniform(-0.02, 0.02)
        margin = 0.05 + rng.uniform(0, 0.02)
        y = margin
        while idx < N and y < 1.0 - margin:
            x_off = margin + (row % 2) * spacing / 2.0
            col = 0
            while x_off + col * spacing < 1.0 - margin and idx < N:
                c[idx] = [x_off + col * spacing, y]
                idx += 1
                col += 1
            y += spacing * np.sqrt(3) / 2.0
            row += 1
        # Fill remaining randomly if lattice doesn't yield 26
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    # 2. Grid patterns with noise
    for seed in range(15):
        rng = np.random.RandomState(100 + seed)
        c = np.zeros((N, 2))
        idx = 0
        step = 0.18 + rng.uniform(-0.02, 0.02)
        for i in range(8):
            for j in range(8):
                if idx < N:
                    c[idx] = [0.05 + j * step, 0.05 + i * step]
                    idx += 1
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        c += rng.normal(0, 0.008, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    # 3. Pure random feasible placements
    for seed in range(20):
        rng = np.random.RandomState(200 + seed)
        c = rng.uniform(0.15, 0.85, (N, 2))
        inits.append(c)
        
    return inits

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0)] * (2 * N) + [(1e-7, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_initializations()
    
    # Phase 1: Broad multi-start search
    for c0 in inits:
        r0, _ = solve_radii_lp(c0)
        # Scale down slightly to guarantee strict feasibility for SLSQP start
        r0 = np.maximum(r0 * 0.96, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, 
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            
            c_opt = res.x[0::3]
            y_opt = res.x[1::3]
            curr_centers = np.column_stack((c_opt, y_opt))
            
            # Phase 2: Exact LP refinement for radii given optimized centers
            r_opt, s_opt = solve_radii_lp(curr_centers)
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = curr_centers.copy()
                best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Phase 3: Local perturbation refinement around the best solution
    if best_centers is not None:
        rng_pert = np.random.RandomState(12345)
        for trial in range(40):
            # Adaptive noise scale
            noise_scale = 0.002 + rng_pert.uniform(0, 0.008) * (trial / 40.0)
            c_pert = best_centers + rng_pert.normal(0, noise_scale, best_centers.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, _ = solve_radii_lp(c_pert)
            r_pert = np.maximum(r_pert * 0.98, 1e-5)
            
            x_pert = np.zeros(3 * N)
            x_pert[0::3] = c_pert[:, 0]
            x_pert[1::3] = c_pert[:, 1]
            x_pert[2::3] = r_pert
            
            try:
                res_p = minimize(objective, x_pert, method='SLSQP', bounds=bounds_opt,
                                 constraints=cons_opt, 
                                 options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                
                c_p_opt = res_p.x[0::3]
                y_p_opt = res_p.x[1::3]
                curr_centers = np.column_stack((c_p_opt, y_p_opt))
                
                r_p_opt, s_p_opt = solve_radii_lp(curr_centers)
                
                if s_p_opt > best_sum:
                    best_sum = s_p_opt
                    best_centers = curr_centers.copy()
                    best_radii = r_p_opt.copy()
            except Exception:
                continue

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # Phase 4: Strict post-processing to guarantee validity
    best_centers, best_radii = make_valid(best_centers, best_radii)
    
    return best_centers, best_radii, float(np.sum(best_radii))

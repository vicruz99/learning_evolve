# sol_000124 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000082 (state 4b2dee7c) state=faf0523f sum of radii=2.629868 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)
NUM_PAIRS = len(I)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I] = 1.0
    A_ub[np.arange(NUM_PAIRS), J] = 1.0
    
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds_r = []
    for i in range(n):
        mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        ub = max(0.0, mx)
        bounds_r.append((0.0, ub))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method=method)
            if res.success:
                return np.maximum(res.x, 0.0), -res.fun
        except Exception:
            continue
    return np.zeros(n), 0.0

def constraints(x):
    """Inequality constraints (must be >= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c_bound = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    dist = np.hypot(dx, dy)
    c_pair = dist - (r[I] + r[J])
    
    return np.concatenate([c_pair, c_bound])

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def make_hex_init(spacing, angle, ox, oy, seed):
    """Generate a perturbed hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    
    v1 = np.array([spacing, 0.0])
    v2 = np.array([spacing * 0.5, spacing * np.sqrt(3) / 2.0])
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    r_range = int(1.0 / (spacing * 0.7)) + 2
    
    for r in range(-r_range, r_range):
        for c in range(-r_range, r_range):
            pt = r * v1 + c * v2
            pt = rot @ pt + np.array([0.5 + ox, 0.5 + oy])
            if 0.05 <= pt[0] <= 0.95 and 0.05 <= pt[1] <= 0.95:
                if idx < N:
                    centers[idx] = pt
                    idx += 1
            if idx >= N:
                break
        if idx >= N:
            break
            
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
        
    centers += rng.normal(0, 0.003, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Diverse structured & random initializations
    inits = []
    base_seed = 42
    for sp in np.linspace(0.13, 0.20, 7):
        for ang in [0.0, np.pi/6, np.pi/4]:
            for ox in [-0.02, 0.0, 0.02]:
                for oy in [-0.02, 0.0, 0.02]:
                    inits.append(make_hex_init(sp, ang, ox, oy, base_seed))
                    
    for seed in range(15):
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0, 1e-4)
        
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0 * 0.92
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            c_opt = np.column_stack((cx, cy))
            r_opt, s_opt = solve_lp_radii(c_opt)
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Basin hopping with force relaxation & SLSQP polish
    if best_centers is not None:
        rng = np.random.RandomState(123)
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_s = best_sum
        
        for step in range(60):
            noise = 0.006 * (0.88 ** (step // 10))
            c_pert = current_c + rng.normal(0, noise, (N, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            # Force-directed relaxation to spread overlapping circles
            for _ in range(25):
                forces = np.zeros((N, 2))
                for i in range(N):
                    for j in range(i+1, N):
                        dx = c_pert[i,0] - c_pert[j,0]
                        dy = c_pert[i,1] - c_pert[j,1]
                        d = np.hypot(dx, dy)
                        if d < current_r[i] + current_r[j] and d > 1e-9:
                            f = (current_r[i] + current_r[j] - d) / d
                            forces[i] += np.array([dx, dy]) * f
                            forces[j] -= np.array([dx, dy]) * f
                c_pert += 0.3 * forces
                c_pert = np.clip(c_pert, 0.02, 0.98)
                
            r_pert, _ = solve_lp_radii(c_pert)
            if np.sum(r_pert) > current_s:
                current_c, current_r, current_s = c_pert, r_pert, np.sum(r_pert)
                
                # Polish with SLSQP
                x0 = np.zeros(3*N)
                x0[0::3] = current_c[:, 0]
                x0[1::3] = current_c[:, 1]
                x0[2::3] = np.maximum(current_r * 0.95, 1e-4)
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    c_pol = np.column_stack((cx, cy))
                    r_pol, s_pol = solve_lp_radii(c_pol)
                    if s_pol > current_s:
                        current_c, current_r, current_s = c_pol, r_pol, s_pol
                except Exception:
                    pass
                    
            if current_s > best_sum:
                best_sum = current_s
                best_centers = current_c.copy()
                best_radii = current_r.copy()
                
    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 3: Strict post-processing to guarantee numerical validity
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    for i in range(N):
        mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(radii[i], 0.0)
        
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc/2.0
                    radii[j] -= exc/2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))

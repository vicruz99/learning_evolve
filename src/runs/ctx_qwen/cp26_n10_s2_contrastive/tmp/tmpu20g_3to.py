import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Constraint matrix structure is constant
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(1e-9, ub)))
        
    # Try HiGHS first (fastest), fallback to interior-point
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success:
                return np.maximum(res.x, 0.0)
        except Exception:
            continue
            
    return np.full(n, 1e-6)

def objective_joint(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def make_strictly_feasible(centers, radii):
    """Deterministically resolve overlaps and boundary violations."""
    for i in range(N):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > ub:
            radii[i] = max(0.0, ub - 1e-9)
            
    for _ in range(50):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < radii[i] + radii[j] - 1e-12:
                exc = radii[i] + radii[j] - d
                radii[i] -= exc * 0.5
                radii[j] -= exc * 0.5
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return centers, radii

def generate_hex_init(rng, scale, margin=0.05, noise=0.005):
    """Generate hexagonal lattice initial configuration."""
    centers = np.zeros((N, 2))
    idx = 0
    y = margin
    row = 0
    while idx < N and y < 1.0 - margin:
        x = margin + (row % 2) * scale / 2.0
        while x < 1.0 - margin and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += scale
        y += scale * np.sqrt(3) / 2.0
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin, 1.0 - margin, 2)
        idx += 1
    centers += rng.normal(0, noise, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Structured & Random Starts with SLSQP
    inits = []
    for s in np.linspace(0.15, 0.25, 15):
        inits.append(generate_hex_init(rng, s, margin=0.04, noise=0.008))
    for _ in range(25):
        inits.append(np.clip(rng.uniform(0.05, 0.95, (N, 2)), 0.02, 0.98))
        
    for c0 in inits:
        r0 = solve_lp_radii(c0) * 0.98
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = np.maximum(r0, 1e-5)
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if res.success or -res.fun > best_sum:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = solve_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Advanced Basin Hopping with Mixed Operators
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(7000):
            progress = step / 7000.0
            temp = 0.012 * np.exp(-progress * 6.0)
            noise = 0.009 * (1.0 - progress * 0.75)
            
            op = rng.choice(['perturb', 'swap', 'rotate'])
            c_new = curr_c.copy()
            
            if op == 'perturb':
                n_pert = rng.integers(1, 5)
                idx = rng.choice(N, n_pert, replace=False)
                c_new[idx] += rng.normal(0, noise, (n_pert, 2))
            elif op == 'swap':
                i, j = rng.choice(N, 2, replace=False)
                c_new[[i, j]] = c_new[[j, i]]
                c_new[[i, j]] += rng.normal(0, noise * 0.4, (2, 2))
            else:
                n_sub = rng.integers(3, 8)
                idx = rng.choice(N, n_sub, replace=False)
                sub_c = c_new[idx]
                centroid = sub_c.mean(axis=0)
                sub_c -= centroid
                angle = rng.uniform(-noise, noise)
                rot = np.array([[np.cos(angle), -np.sin(angle)], 
                                [np.sin(angle), np.cos(angle)]])
                c_new[idx] = sub_c @ rot.T + centroid
                
            c_new = np.clip(c_new, 0.01, 0.99)
            
            r_new = solve_lp_radii(c_new)
            s_new = np.sum(r_new)
            
            accept = False
            if s_new > curr_s:
                accept = True
            elif temp > 1e-6 and s_new > 0:
                if rng.random() < np.exp((s_new - curr_s) / max(temp, 1e-4)):
                    accept = True
                    
            if accept:
                curr_c = c_new
                curr_r = r_new
                curr_s = s_new
                
                if s_new > best_sum:
                    best_sum = s_new
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
                    
                    # Periodic SLSQP polishing
                    if step % 400 == 0:
                        x0_p = np.zeros(3 * N)
                        x0_p[0::3] = best_centers[:, 0]
                        x0_p[1::3] = best_centers[:, 1]
                        x0_p[2::3] = np.maximum(best_radii * 0.99, 1e-5)
                        try:
                            res_p = minimize(objective_joint, x0_p, method='SLSQP', bounds=bounds_opt,
                                             constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                            if res_p.success:
                                c_opt = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                                r_opt = solve_lp_radii(c_opt)
                                if np.sum(r_opt) > best_sum:
                                    best_sum = np.sum(r_opt)
                                    best_centers = c_opt.copy()
                                    best_radii = r_opt.copy()
                                    curr_c, curr_r, curr_s = best_centers, best_radii, best_sum
                        except Exception:
                            pass

    # Phase 3: Final Intense Polishing
    if best_centers is not None:
        c_final = best_centers.copy()
        r_final = best_radii.copy()
        for _ in range(8):
            x0 = np.zeros(3 * N)
            x0[0::3] = c_final[:, 0]
            x0[1::3] = c_final[:, 1]
            x0[2::3] = np.maximum(r_final * 0.995, 1e-5)
            try:
                res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                               constraints=cons_opt, options={'maxiter': 15000, 'ftol': 1e-15, 'disp': False})
                if res.success:
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_opt = solve_lp_radii(c_opt)
                    if np.sum(r_opt) > np.sum(r_final):
                        c_final, r_final = c_opt, r_opt
            except Exception:
                pass
        best_centers, best_radii = c_final, r_final

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii = solve_lp_radii(best_centers)
        
    # Strict post-processing to guarantee validator compliance
    best_centers, best_radii = make_strictly_feasible(best_centers.copy(), best_radii.copy())
    
    # Final boundary clamp
    for i in range(N):
        ub = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        best_radii[i] = min(best_radii[i], ub - 1e-9)
        best_radii[i] = max(0.0, best_radii[i])
        
    return best_centers, best_radii, float(np.sum(best_radii))
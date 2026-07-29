import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-10, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-4), 0.0

def objective_joint(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
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
    # Squared distance constraints avoid gradient singularities at contact
    c[4*N:] = dx**2 + dy**2 - (r[I_IDX] + r[J_IDX])**2
    return c

bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
cons_opt = {'type': 'ineq', 'fun': constraints_joint}

def generate_hex_init(spacing, margin=0.0, shift_x=0.0, shift_y=0.0):
    """Generate a hexagonal lattice initialization."""
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin + spacing/2 + shift_y
    while idx < N and y < 1.0 - margin - spacing/2:
        x_start = margin + spacing/2 + shift_x + (row % 2) * spacing/2
        col = 0
        while x_start + col * spacing < 1.0 - margin - spacing/2 and idx < N:
            centers[idx, 0] = x_start + col * spacing
            centers[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = [0.5, 0.5]
        idx += 1
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Generate diverse starts
    starts = []
    for s in np.linspace(0.16, 0.24, 10):
        for mx in [0.02, 0.05, 0.08]:
            starts.append(generate_hex_init(s, margin=mx))
            
    for _ in range(30):
        starts.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    for seed in range(20):
        r_s = np.random.RandomState(seed)
        c = generate_hex_init(0.18 + r_s.uniform(-0.02, 0.02))
        c += r_s.normal(0, 0.015, c.shape)
        starts.append(np.clip(c, 0.02, 0.98))
        
    # Phase 2: Initial SLSQP polish from diverse starts
    for c0 in starts:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0 * 0.95, 1e-5)
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                curr_c = np.column_stack((cx, cy))
                curr_r, curr_s = solve_lp_radii(curr_c)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = starts[0]
        best_radii, best_sum = solve_lp_radii(best_centers)

    # Phase 3: Deterministic Hill-Climbing on Centers maximizing LP objective
    curr_c = best_centers.copy()
    curr_r = best_radii.copy()
    curr_s = best_sum
    scale = 0.015
    
    for step in range(800):
        improved = False
        n_pert = rng.integers(1, 7)
        idxs = rng.choice(N, n_pert, replace=False)
        
        pert_c = curr_c.copy()
        pert_c[idxs] += rng.normal(0, scale, (n_pert, 2))
        pert_c = np.clip(pert_c, 0.005, 0.995)
        
        r_new, s_new = solve_lp_radii(pert_c)
        if s_new > curr_s:
            curr_c, curr_r, curr_s = pert_c, r_new, s_new
            improved = True
            scale *= 1.1
        else:
            scale *= 0.96
            
        scale = min(scale, 0.04)
        if scale < 1e-5:
            break
            
        if curr_s > best_sum:
            best_sum = curr_s
            best_centers = curr_c.copy()
            best_radii = curr_r.copy()
            
            # Occasional SLSQP polish during hill climb to tighten configuration
            if step % 60 == 0:
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = best_radii * 0.98
                try:
                    res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                    if res.success:
                        cc = np.column_stack((res.x[0::3], res.x[1::3]))
                        cr, cs = solve_lp_radii(cc)
                        if cs > best_sum:
                            best_sum = cs
                            best_centers = cc
                            best_radii = cr
                            curr_c, curr_r, curr_s = cc, cr, cs
                except Exception:
                    pass
                    
    # Phase 4: Basin hopping restarts to escape local minima
    for _ in range(5):
        c_restart = best_centers + rng.normal(0, 0.03, best_centers.shape)
        c_restart = np.clip(c_restart, 0.02, 0.98)
        r_restart, s_restart = solve_lp_radii(c_restart)
        if s_restart > best_sum:
            best_sum = s_restart
            best_centers = c_restart.copy()
            best_radii = r_restart.copy()
            curr_c, curr_r, curr_s = c_restart, r_restart, s_restart
            
            # Local hill climb from restart
            scale = 0.01
            for _ in range(200):
                n_pert = rng.integers(1, 5)
                idxs = rng.choice(N, n_pert, replace=False)
                pert_c = curr_c.copy()
                pert_c[idxs] += rng.normal(0, scale, (n_pert, 2))
                pert_c = np.clip(pert_c, 0.005, 0.995)
                r_new, s_new = solve_lp_radii(pert_c)
                if s_new > curr_s:
                    curr_c, curr_r, curr_s = pert_c, r_new, s_new
                    scale *= 1.05
                else:
                    scale *= 0.95
                scale = min(scale, 0.03)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()

    # Phase 5: Final intensive SLSQP polish
    x0 = np.zeros(3*N)
    x0[0::3] = best_centers[:, 0]
    x0[1::3] = best_centers[:, 1]
    x0[2::3] = best_radii * 0.99
    try:
        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                       constraints=cons_opt, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        if res.success:
            cc = np.column_stack((res.x[0::3], res.x[1::3]))
            cr, cs = solve_lp_radii(cc)
            if cs > best_sum:
                best_sum = cs
                best_centers = cc
                best_radii = cr
    except Exception:
        pass
        
    # Phase 6: Strict post-processing to guarantee validator compliance
    centers_out = best_centers.copy()
    radii_out = best_radii.copy()
    
    for i in range(N):
        mx = min(centers_out[i, 0], 1.0 - centers_out[i, 0], 
                 centers_out[i, 1], 1.0 - centers_out[i, 1])
        radii_out[i] = min(radii_out[i], max(0.0, mx - 1e-9))
        
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(centers_out[i,0]-centers_out[j,0], centers_out[i,1]-centers_out[j,1])
            if d < radii_out[i] + radii_out[j] - 1e-12:
                exc = radii_out[i] + radii_out[j] - d
                radii_out[i] -= exc * 0.5
                radii_out[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    radii_out = np.maximum(radii_out, 0.0)
    return centers_out, radii_out, float(np.sum(radii_out))
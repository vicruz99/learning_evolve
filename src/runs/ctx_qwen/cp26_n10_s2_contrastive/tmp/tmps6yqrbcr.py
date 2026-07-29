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
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def relax_centers(centers, radii, steps=40):
    """Deterministically push overlapping circles apart to improve packing density."""
    c = centers.copy()
    n = c.shape[0]
    for _ in range(steps):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                if d < radii[i] + radii[j] and d > 1e-9:
                    shift = (radii[i] + radii[j] - d) * 0.5
                    ux = dx / d
                    uy = dy / d
                    c[i, 0] += ux * shift
                    c[i, 1] += uy * shift
                    c[j, 0] -= ux * shift
                    c[j, 1] -= uy * shift
                    moved = True
        c = np.clip(c, 0.001, 0.999)
        if not moved:
            break
    return c

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
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

def generate_inits(rng):
    """Generate diverse structured initial configurations."""
    inits = []
    # 1. Hexagonal lattices with varying density and alignment
    for seed in range(40):
        r_gen = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        s = 0.13 + r_gen.uniform(0, 0.08)
        y = s / 2
        row = 0
        while idx < N and y < 1.0 - s / 2:
            x_start = s / 2 + (row % 2) * s / 2
            col = 0
            while x_start + col * s < 1.0 - s / 2 and idx < N:
                c[idx] = [x_start + col * s, y]
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(0.1, 0.9, 2)
            idx += 1
        c += r_gen.normal(0, 0.008, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # 2. Random uniform placements
    for _ in range(30):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # 3. Corner-biased layouts
    for _ in range(20):
        c = rng.uniform(0.1, 0.9, (N, 2))
        corners = rng.uniform(0.02, 0.15, (4, 2))
        c[:4] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) + corners
        inits.append(np.clip(c, 0.01, 0.99))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start SLSQP with LP initialization
    for c0 in inits:
        r0 = solve_lp_radii(c0) * 0.95
        r0 = np.maximum(r0, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                curr_c = np.column_stack((cx, cy))
                curr_r = solve_lp_radii(curr_c)
                curr_s = np.sum(curr_r)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            pass

    # Phase 2: Alternating Relaxation + LP refinement
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for _ in range(5):
            # Relax to escape tight local minima
            curr_c = relax_centers(curr_c, curr_r, steps=50)
            curr_r = solve_lp_radii(curr_c)
            curr_s = np.sum(curr_r)
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                
            # Quick SLSQP polish after relaxation
            x0 = np.zeros(3 * N)
            x0[0::3] = curr_c[:, 0]
            x0[1::3] = curr_c[:, 1]
            x0[2::3] = np.maximum(curr_r * 0.98, 1e-5)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                               constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                if res.success:
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_opt = solve_lp_radii(c_opt)
                    s_opt = np.sum(r_opt)
                    if s_opt > best_sum:
                        best_sum = s_opt
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
            except Exception:
                pass

    # Phase 3: Targeted Hill-Climbing on Centers maximizing LP Radii Sum
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        step = 0.015
        for it in range(200):
            improved = False
            # Try multiple targeted perturbations per iteration
            for _ in range(60):
                pert = curr_c.copy()
                n_pert = rng.integers(1, 6)
                idx_pert = rng.choice(N, n_pert, replace=False)
                pert[idx_pert] += rng.normal(0, step, (n_pert, 2))
                pert = np.clip(pert, 0.01, 0.99)
                
                # Light relaxation helps LP evaluate better
                pert = relax_centers(pert, curr_r, steps=15)
                r_pert = solve_lp_radii(pert)
                s_pert = np.sum(r_pert)
                
                if s_pert > curr_s:
                    curr_c, curr_r, curr_s = pert, r_pert, s_pert
                    improved = True
                    if curr_s > best_sum:
                        best_sum = curr_s
                        best_centers = curr_c.copy()
                        best_radii = curr_r.copy()
                        
            if not improved:
                step *= 0.85
            else:
                step *= 1.05
                step = min(step, 0.03)
            if step < 1e-5:
                break
                
    # Phase 4: Strict post-processing to guarantee validator compliance
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        best_radii[i] = min(best_radii[i], max(0.0, mx - 1e-9))
        
    for _ in range(50):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            if d < best_radii[i] + best_radii[j] - 1e-11:
                exc = best_radii[i] + best_radii[j] - d
                best_radii[i] -= exc * 0.5
                best_radii[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(np.sum(best_radii))
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2 * N] = 1.0 - cx - r
    c[2 * N:3 * N] = cy - r
    c[3 * N:4 * N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4 * N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

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
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def relax_centers_force(centers, max_iter=400):
    """Force-directed relaxation to quickly resolve overlaps and pack circles densely."""
    c = centers.copy()
    n = c.shape[0]
    k_rep = 4.0
    dt = 0.008
    
    for _ in range(max_iter):
        forces = np.zeros_like(c)
        # Vectorized pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        np.fill_diagonal(dists, 1.0)
        
        mask = dists < 0.25
        inv_dist = np.where(mask & (dists > 1e-6), 1.0 / dists, 0.0)
        force_magnitude = np.where(mask, k_rep * (0.25 - dists) * inv_dist, 0.0)
        
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    fx = force_magnitude[i, j] * diff[i, j, 0]
                    fy = force_magnitude[i, j] * diff[i, j, 1]
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
                    
        c += forces * dt
        c = np.clip(c, 1e-4, 1.0 - 1e-4)
    return c

def make_strictly_feasible(centers, radii):
    """Deterministically resolve overlaps and boundary violations."""
    for i in range(N):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > ub:
            radii[i] = max(0.0, ub - 1e-9)
            
    for _ in range(100):
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

def generate_patterns():
    """Generate diverse initial configurations tailored for N=26."""
    inits = []
    rng = np.random.default_rng(123)
    
    # 1. Hexagonal lattices with varying spacings and y-shifts
    for sp in np.linspace(0.15, 0.25, 12):
        for sy in [0.0, 0.04, 0.08]:
            c = np.zeros((N, 2))
            idx = 0
            y = 0.05 + sy
            row = 0
            while idx < N and y < 0.95:
                x = 0.05 + (row % 2) * sp / 2.0
                while x < 0.95 and idx < N:
                    c[idx] = [x, y]
                    idx += 1
                    x += sp
                y += sp * np.sqrt(3) / 2.0
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            inits.append(c + rng.normal(0, 0.006, c.shape))
            
    # 2. Specific row count patterns for 26 (truncated hexagons)
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], 
                [8,5,6,5,2], [6,6,6,4,4], [5,5,5,5,6], [4,6,5,6,5]]
    for pat in patterns:
        c = np.zeros((N, 2))
        idx = 0
        y = 0.08
        dy = 0.84 / (len(pat) - 0.5)
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.09
            x = 0.06 + shift
            if cnt > 1:
                spacing = (0.88 - 2*shift) / (cnt - 1)
            else:
                spacing = 0.25
            for _ in range(cnt):
                if idx < N:
                    c[idx] = [x, y]
                    idx += 1
                x += spacing
            y += dy
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        inits.append(c + rng.normal(0, 0.007, c.shape))
        
    # 3. Random uniform placements
    for _ in range(35):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_patterns()
    
    # Phase 1: Multi-start SLSQP with force relaxation
    for base in inits:
        c_init = np.clip(base, 0.02, 0.98)
        c_relaxed = relax_centers_force(c_init, max_iter=300)
        r_init = solve_lp_radii(c_relaxed) * 0.97
        x0 = np.zeros(3 * N)
        x0[0::3] = c_relaxed[:, 0]
        x0[1::3] = c_relaxed[:, 1]
        x0[2::3] = np.maximum(r_init, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt,
                           options={'maxiter': 18000, 'ftol': 1e-14, 'disp': False})
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

    # Phase 2: Adaptive Basin Hopping
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(4000):
            scale = 0.009 * np.exp(-step / 1200.0)
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.015, 0.985)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                
                # Polish successful jump
                x0_p = np.zeros(3 * N)
                x0_p[0::3] = curr_c[:, 0]
                x0_p[1::3] = curr_c[:, 1]
                x0_p[2::3] = np.maximum(curr_r * 0.99, 1e-5)
                try:
                    res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt,
                                     options={'maxiter': 9000, 'ftol': 1e-14, 'disp': False})
                    if res_p.success:
                        c_opt = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                        r_opt = solve_lp_radii(c_opt)
                        s_opt = np.sum(r_opt)
                        if s_opt > curr_s:
                            curr_c, curr_r, curr_s = c_opt, r_opt, s_opt
                            best_sum = curr_s
                            best_centers = curr_c.copy()
                            best_radii = curr_r.copy()
                except Exception:
                    pass
                    
    # Phase 3: Targeted Individual/Small-Group Perturbations
    if best_centers is not None:
        for _ in range(1000):
            num_pert = rng.choice([1, 2, 3])
            idxs = rng.choice(N, num_pert, replace=False)
            c_pert = best_centers.copy()
            noise = rng.uniform(0.0015, 0.006)
            c_pert[idxs] += rng.normal(0, noise, (num_pert, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                
                # Quick polish
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 7000, 'ftol': 1e-14, 'disp': False})
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_opt = solve_lp_radii(c_opt)
                    if np.sum(r_opt) > best_sum:
                        best_sum = np.sum(r_opt)
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    c_final, r_final = make_strictly_feasible(best_centers.copy(), best_radii.copy())
    
    for i in range(N):
        ub = min(c_final[i, 0], 1.0 - c_final[i, 0], c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], ub - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    return c_final, r_final, float(np.sum(r_final))
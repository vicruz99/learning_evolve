import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)
A_LP = np.zeros((NUM_PAIRS, N))
A_LP[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_LP[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0-x, y, 1.0-y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 1e-6)

def objective_joint(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx, cy, r = x[0::3], x[1::3], x[2::3]
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    dists = np.hypot(dx, dy)
    overlap = dists - (r[I_IDX] + r[J_IDX])
    bound = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([overlap, bound])

def push_apart(centers, radii, steps=25):
    """Force-directed relaxation to separate overlapping circles."""
    c = centers.copy()
    r = radii.copy()
    for _ in range(steps):
        moves = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-12:
                    overlap = r[i] + r[j] - d
                    shift = overlap / 2.0
                    ux, uy = dx / d, dy / d
                    moves[i] += np.array([ux * shift, uy * shift])
                    moves[j] -= np.array([ux * shift, uy * shift])
        c += moves * 0.4
        c = np.clip(c, 0.015, 0.985)
    return c

def generate_inits(rng):
    """Generate diverse structured initial configurations."""
    inits = []
    # Hexagonal lattices with varying spacing
    for s in np.linspace(0.14, 0.23, 18):
        c = np.zeros((N, 2))
        idx = 0
        y = s / 2
        row = 0
        while idx < N and y < 1.0 - s / 2:
            x = s / 2 + (row % 2) * s / 2
            while x < 1.0 - s / 2 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = rng.uniform(0.2, 0.8, 2)
            idx += 1
        inits.append(c + rng.normal(0, 0.006, c.shape))
        
    # Random uniform placements
    for _ in range(35):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    inits = generate_inits(rng)
    
    # Phase 1: Adaptive Hill-Climbing on Centers maximizing LP Radii Sum
    for c0 in inits:
        c_curr = np.clip(c0, 0.02, 0.98)
        r_curr = solve_lp_radii(c_curr)
        s_curr = np.sum(r_curr)
        
        step = 0.018
        for it in range(70):
            improved = False
            for _ in range(25):
                c_pert = c_curr + rng.normal(0, step, (N, 2))
                c_pert = np.clip(c_pert, 0.02, 0.98)
                r_pert = solve_lp_radii(c_pert)
                s_pert = np.sum(r_pert)
                if s_pert > s_curr:
                    c_curr, r_curr, s_curr = c_pert, r_pert, s_pert
                    improved = True
            if not improved:
                step *= 0.82
            else:
                step = min(step * 1.15, 0.035)
                
        if s_curr > best_sum:
            best_sum = s_curr
            best_centers = c_curr.copy()
            best_radii = r_curr.copy()
            
        # Force relaxation to find new basins
        c_pushed = push_apart(c_curr, r_curr, steps=30)
        r_pushed = solve_lp_radii(c_pushed)
        s_pushed = np.sum(r_pushed)
        if s_pushed > best_sum:
            best_sum = s_pushed
            best_centers = c_pushed.copy()
            best_radii = r_pushed.copy()

    # Phase 2: Joint SLSQP Refinement
    if best_centers is not None:
        c_pol = best_centers.copy()
        r_pol = best_radii.copy()
        x0 = np.zeros(3 * N)
        x0[0::3] = c_pol[:, 0]
        x0[1::3] = c_pol[:, 1]
        x0[2::3] = np.maximum(r_pol * 0.97, 1e-5)
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = solve_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt
                    best_radii = r_opt
        except Exception:
            pass
            
        # Basin Hopping around best configuration
        for _ in range(40):
            scale = rng.uniform(0.002, 0.012)
            c_hop = best_centers + rng.normal(0, scale, (N, 2))
            c_hop = np.clip(c_hop, 0.02, 0.98)
            r_hop = solve_lp_radii(c_hop)
            s_hop = np.sum(r_hop)
            if s_hop > best_sum:
                best_sum = s_hop
                best_centers = c_hop
                best_radii = r_hop
                
                # Quick SLSQP polish after successful jump
                x0h = np.zeros(3 * N)
                x0h[0::3] = c_hop[:, 0]
                x0h[1::3] = c_hop[:, 1]
                x0h[2::3] = np.maximum(r_hop * 0.98, 1e-5)
                try:
                    res_h = minimize(objective_joint, x0h, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                    if res_h.success:
                        c_h = np.column_stack((res_h.x[0::3], res_h.x[1::3]))
                        r_h = solve_lp_radii(c_h)
                        s_h = np.sum(r_h)
                        if s_h > best_sum:
                            best_sum = s_h
                            best_centers = c_h
                            best_radii = r_h
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        best_centers = np.random.uniform(0.2, 0.8, (N, 2))
        best_radii = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Strict post-processing to guarantee validator compliance
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        best_radii[i] = min(best_radii[i], max(0.0, mx - 1e-9))
        
    for _ in range(150):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            if d < best_radii[i] + best_radii[j] - 1e-11:
                exc = best_radii[i] + best_radii[j] - d
                best_radii[i] -= exc * 0.5
                best_radii[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(np.sum(best_radii))
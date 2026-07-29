import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute constant LP structure for speed
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 1e-6)

def neg_sum_radii_flat(x_flat):
    """Objective for center-only optimization: minimize negative sum of LP radii."""
    centers = x_flat.reshape(N, 2)
    centers = np.clip(centers, 0.001, 0.999)
    radii = solve_lp_radii(centers)
    return -np.sum(radii)

def objective_joint(x):
    """Joint objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Joint inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_overlap = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    return np.concatenate([c_overlap, c_bound])

def generate_inits():
    """Generate diverse initial center configurations."""
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Hexagonal lattices with varying spacing
    for sp in np.linspace(0.16, 0.22, 10):
        c = np.zeros((N, 2))
        idx = 0; row = 0; y = sp/2
        while idx < N and y < 1.0 - sp/2:
            x = sp/2 + (row%2)*sp/2
            while x < 1.0 - sp/2 and idx < N:
                c[idx] = [x, y]; idx += 1; x += sp
            y += sp*np.sqrt(3)/2; row += 1
        while idx < N: c[idx] = rng.uniform(0.1, 0.9, 2); idx += 1
        inits.append(c + rng.normal(0, 0.005, c.shape))
        
    # 2. Square grids
    for sp in np.linspace(0.17, 0.21, 8):
        c = np.zeros((N, 2))
        idx = 0; y = sp/2
        while y < 1.0 - sp/2 and idx < N:
            x = sp/2
            while x < 1.0 - sp/2 and idx < N:
                c[idx] = [x, y]; idx += 1; x += sp
            y += sp
        while idx < N: c[idx] = rng.uniform(0.1, 0.9, 2); idx += 1
        inits.append(c + rng.normal(0, 0.005, c.shape))
        
    # 3. Corner and edge focused layouts
    for _ in range(8):
        c = np.zeros((N, 2))
        c[0] = [0.1, 0.1]; c[1] = [0.9, 0.1]; c[2] = [0.1, 0.9]; c[3] = [0.9, 0.9]
        pts = np.linspace(0.15, 0.85, 5)
        idx = 4
        for x in pts:
            if idx < N: c[idx] = [x, 0.1]; idx += 1
        for x in pts:
            if idx < N: c[idx] = [x, 0.9]; idx += 1
        for y in pts:
            if idx < N: c[idx] = [0.1, y]; idx += 1
        for y in pts:
            if idx < N: c[idx] = [0.9, y]; idx += 1
        while idx < N: c[idx] = rng.uniform(0.25, 0.75, 2); idx += 1
        c += rng.normal(0, 0.008, c.shape)
        inits.append(np.clip(c, 0.05, 0.95))
        
    # 4. Specific row patterns matching N=26
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6]]
    for pat in patterns:
        c = np.zeros((N, 2))
        idx = 0
        y = 0.08
        dy = 0.84 / (len(pat) - 0.5)
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.09
            x = 0.08 + shift
            for _ in range(cnt):
                if idx < N:
                    c[idx] = [x, y]
                    idx += 1
                x += (0.84 - 2*shift) / (cnt - 0.5) if cnt > 1 else 0.0
            y += dy
        while idx < N: c[idx] = rng.uniform(0.2, 0.8, 2); idx += 1
        inits.append(c + rng.normal(0, 0.005, c.shape))
        
    # 5. Random uniform
    for _ in range(12):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_inits()
    rng = np.random.default_rng(2024)
    
    # Phase 1: Center-only optimization with Nelder-Mead + LP evaluation
    for c0 in inits:
        c0 = np.clip(c0, 0.02, 0.98)
        try:
            res_c = minimize(neg_sum_radii_flat, c0.flatten(), method='Nelder-Mead',
                             options={'maxiter': 2500, 'fatol': 1e-12})
            c_opt = np.clip(res_c.x.reshape(N, 2), 0.02, 0.98)
            r_opt = solve_lp_radii(c_opt)
            s_opt = np.sum(r_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Joint SLSQP refinement for precise constraint satisfaction
    if best_centers is not None:
        x0 = np.zeros(3*N)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = best_radii * 0.95
        
        try:
            res_j = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                             constraints=cons_opt, options={'maxiter': 12000, 'ftol': 1e-14})
            cx = res_j.x[0::3]; cy = res_j.x[1::3]
            co = np.column_stack((cx, cy))
            ro = solve_lp_radii(co)
            so = np.sum(ro)
            if so > best_sum:
                best_sum = so
                best_centers = co.copy()
                best_radii = ro.copy()
        except Exception:
            pass
            
    # Phase 3: Basin Hopping on Centers with adaptive cooling
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_s = best_sum
        
        for step in range(600):
            noise = 0.014 * np.exp(-step / 150.0) + 0.0008
            c_pert = curr_c + rng.normal(0, noise, (N, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > curr_s:
                curr_c = c_pert
                curr_s = s_pert
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_centers = c_pert.copy()
                    best_radii = r_pert.copy()
                    
                    # Local polish after successful jump
                    try:
                        res_p = minimize(neg_sum_radii_flat, c_pert.flatten(), method='Nelder-Mead',
                                         options={'maxiter': 1000, 'fatol': 1e-12})
                        co_p = np.clip(res_p.x.reshape(N, 2), 0.02, 0.98)
                        ro_p = solve_lp_radii(co_p)
                        so_p = np.sum(ro_p)
                        if so_p > best_sum:
                            best_sum = so_p
                            best_centers = co_p.copy()
                            best_radii = ro_p.copy()
                            curr_c = co_p
                            curr_s = so_p
                    except Exception:
                        pass
                        
    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(radii[i], 0.0)
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-9:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
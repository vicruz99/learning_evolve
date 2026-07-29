# sol_000114 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000091 (state 4dfa0868) state=5134eb3f sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure (r_i + r_j <= dist_ij)
A_LP = np.zeros((NUM_PAIRS, N))
A_LP[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_LP[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Compute pairwise distances
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    # Boundary bounds for each radius
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.01)

def objective_joint(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
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

def generate_structured_inits(rng):
    """Generate diverse hexagonal and grid patterns."""
    inits = []
    # Row count patterns summing to 26
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 5, 4, 4],
        [4, 5, 6, 5, 6], [8, 5, 6, 4, 3], [5, 5, 5, 5, 6],
        [7, 7, 5, 4, 3], [6, 6, 6, 5, 3], [9, 6, 5, 4, 2]
    ]
    
    for pat in patterns:
        for s in np.linspace(0.15, 0.20, 5):
            pts = []
            y = s * 0.65
            dy = s * np.sqrt(3) / 2 * 0.98
            for r_idx, cnt in enumerate(pat):
                shift = 0.0 if r_idx % 2 == 0 else s / 2.0
                x = s * 0.65 + shift
                for _ in range(cnt):
                    if len(pts) < N:
                        pts.append([x, y])
                    x += s
                y += dy
            while len(pts) < N:
                pts.append([0.5, 0.5])
            pts = np.array(pts[:N])
            pts += rng.normal(0, 0.005, pts.shape)
            inits.append(np.clip(pts, 0.03, 0.97))
            
    # Random uniform
    for _ in range(15):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    inits = generate_structured_inits(rng)
    
    # Phase 1: Multi-start SLSQP from structured seeds
    for base in inits:
        r_init = solve_radii_lp(base) * 0.96
        x0 = np.zeros(3 * N)
        x0[0::3] = base[:, 0]
        x0[1::3] = base[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                curr_r = solve_radii_lp(curr_c)
                curr_s = np.sum(curr_r)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            pass

    if best_centers is None:
        best_centers = inits[0]
        best_radii = solve_radii_lp(best_centers)
        best_sum = np.sum(best_radii)

    # Phase 2: Adaptive Basin Hopping & Joint Refinement
    current_c = best_centers.copy()
    current_r = best_radii.copy()
    current_s = best_sum
    
    for step in range(400):
        # Decaying noise schedule
        scale = 0.012 * np.exp(-step / 120.0) + 0.0005
        
        # Random center perturbation
        c_pert = current_c + rng.normal(0, scale, current_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        # Occasionally swap two circles to rearrange local topology
        if rng.random() < 0.15:
            i, j = rng.choice(N, 2, replace=False)
            c_pert[[i, j]] = c_pert[[j, i]]
            
        r_pert = solve_radii_lp(c_pert)
        s_pert = np.sum(r_pert)
        
        if s_pert > current_s:
            current_c, current_r, current_s = c_pert, r_pert, s_pert
            
            # Polish with SLSQP after successful jump
            x0 = np.zeros(3 * N)
            x0[0::3] = current_c[:, 0]
            x0[1::3] = current_c[:, 1]
            x0[2::3] = current_r * 0.97
            
            try:
                res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                               constraints=cons_opt, options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
                if res.success:
                    c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_pol = solve_radii_lp(c_pol)
                    s_pol = np.sum(r_pol)
                    if s_pol > current_s:
                        current_c, current_r, current_s = c_pol, r_pol, s_pol
            except Exception:
                pass
                
            if current_s > best_sum:
                best_sum = current_s
                best_centers = current_c.copy()
                best_radii = current_r.copy()

    # Phase 3: Fine-tuning subset perturbations
    for _ in range(150):
        subset_size = rng.integers(3, 8)
        idx = rng.choice(N, subset_size, replace=False)
        c_pert = best_centers.copy()
        c_pert[idx] += rng.normal(0, 0.002, (subset_size, 2))
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        r_pert = solve_radii_lp(c_pert)
        s_pert = np.sum(r_pert)
        
        if s_pert > best_sum:
            best_sum = s_pert
            best_centers = c_pert.copy()
            best_radii = r_pert.copy()

    # Phase 4: Strict post-processing to guarantee validator compliance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed:
            break
            
    return c_final, r_final, float(np.sum(r_final))

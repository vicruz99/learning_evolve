import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = np.zeros((N_PAIRS, n))
    A_ub[np.arange(N_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(N_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Vectorized pairwise distance constraints
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_dist = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_b = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_dist, c_b])

def rotate_centers(centers, angle):
    """Rotate configuration and re-center to keep inside [0,1]^2."""
    c = centers.copy()
    cx, cy = c[:, 0], c[:, 1]
    new_cx = cx * np.cos(angle) - cy * np.sin(angle)
    new_cy = cx * np.sin(angle) + cy * np.cos(angle)
    new_c = np.column_stack((new_cx, new_cy))
    new_c -= new_c.min(axis=0)
    new_c /= np.maximum(new_c.max(axis=0), 1e-12)
    new_c *= 0.8
    new_c += 0.1
    return np.clip(new_c, 0.02, 0.98)

def generate_hex_init(spacing, seed):
    """Generate a hexagonal lattice initialization with controlled noise."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = spacing / 2
    while idx < N and y < 1.0 - spacing / 2:
        x_start = spacing / 2 + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - spacing / 2 and idx < N:
            centers[idx, 0] = x_start + col * spacing
            centers[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def make_strictly_feasible(centers, radii):
    """Deterministically resolve overlaps and boundary violations."""
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-9)
            
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-12:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Diverse Initializations
    inits = []
    for sp in np.linspace(0.14, 0.22, 10):
        for seed in range(5):
            inits.append(generate_hex_init(sp, seed))
            
    for seed in range(30):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = np.maximum(r0 * 0.95, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt, s_opt = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Basin Hopping with Rotations & Perturbations
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(150):
            # Adaptive noise schedule
            scale = 0.012 * np.exp(-step / 40.0)
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            # Occasional rotation to break symmetry and escape local traps
            if step % 20 == 0:
                angle = rng.uniform(0.05, 0.3)
                c_pert = rotate_centers(curr_c, angle)
                
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > 0:
                x0 = np.zeros(3*N)
                x0[0::3] = c_pert[:, 0]
                x0[1::3] = c_pert[:, 1]
                x0[2::3] = r_pert
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                   options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                    c_new = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new, s_new = solve_lp_radii(c_new)
                    
                    if s_new > curr_s:
                        curr_s = s_new
                        curr_c = c_new.copy()
                        curr_r = r_new.copy()
                        if s_new > best_sum:
                            best_sum = s_new
                            best_centers = c_new.copy()
                            best_radii = r_new.copy()
                except Exception:
                    pass
                    
    # Phase 3: Micro-adjustments & Individual perturbations
    if best_centers is not None:
        for _ in range(80):
            # Perturb 1-4 random circles to fine-tune bottlenecks
            num_p = rng.randint(1, 5)
            idxs = rng.choice(N, num_p, replace=False)
            c_pert = best_centers.copy()
            noise = rng.uniform(0.001, 0.006)
            c_pert[idxs] += rng.normal(0, noise, (num_p, 2))
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.98, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                   options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    c_f = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_f, s_f = solve_lp_radii(c_f)
                    if s_f > best_sum:
                        best_sum = s_f
                        best_centers = c_f.copy()
                        best_radii = r_f.copy()
                except Exception:
                    pass
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = generate_hex_init(0.17, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Strict post-processing to guarantee validator compliance
    best_centers, best_radii = make_strictly_feasible(best_centers.copy(), best_radii.copy())
    return best_centers, best_radii, float(np.sum(best_radii))
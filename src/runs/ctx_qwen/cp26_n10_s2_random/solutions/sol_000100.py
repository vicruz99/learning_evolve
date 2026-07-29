# sol_000100 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000069 (state 2c1a60b6) state=3924d289 sum of radii=2.621719 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_COUNT = N * (N - 1) // 2
i_idx, j_idx = np.triu_indices(N, k=1)

def get_bounds():
    """Creates variable bounds: x,y in [0,1], r in [1e-7, 0.5]"""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def objective(p):
    """Minimize negative sum of radii to maximize total radius."""
    return -np.sum(p[2::3])

def constraints(p):
    """
    Computes all boundary and non-overlap constraints.
    Returns a 1D array where each element must be >= 0.
    Uses squared distances for better gradient conditioning.
    """
    x, y, r = p[0::3], p[1::3], p[2::3]
    c = np.empty(4 * N + PAIR_COUNT)
    idx = 0
    
    # Boundary constraints
    c[idx:idx+N] = x - r; idx += N
    c[idx:idx+N] = 1.0 - x - r; idx += N
    c[idx:idx+N] = y - r; idx += N
    c[idx:idx+N] = 1.0 - y - r; idx += N
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dr = r[i_idx] + r[j_idx]
    c[idx:] = dx*dx + dy*dy - dr*dr
    return c

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to find maximal radii satisfying all constraints."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    num_constraints = PAIR_COUNT + 4 * n
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    for i in range(n):
        A_ub[idx, i] = 1.0; b_ub[idx] = x[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y[i]; idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, method='highs')
    if res.success:
        return res.x
    return np.full(n, 1e-7)

def run_slsqp(p0, maxiter=5000):
    """Runs SLSQP optimization and returns optimized parameters and success flag."""
    try:
        res = minimize(objective, p0, method='SLSQP', bounds=get_bounds(),
                       constraints={'type': 'ineq', 'fun': constraints},
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'disp': False})
        return res.x, res.success
    except Exception:
        return p0, False

def generate_inits():
    """Generates a diverse set of initial configurations."""
    inits = []
    np.random.seed(42)
    
    # 1. Hexagonal lattices with varying noise
    for seed in range(20):
        noise = 0.003 + seed * 0.001
        centers = np.zeros((N, 2))
        idx = 0
        y = 0.09
        row = 0
        r = 0.09
        while idx < N:
            x_start = r if row % 2 == 0 else 2 * r
            x = x_start
            while x + r <= 1.0 + 1e-9 and idx < N:
                centers[idx] = [x, y]
                idx += 1
                x += 2 * r
            y += np.sqrt(3) * r
            row += 1
        centers += np.random.normal(0, noise, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        p = np.zeros(N * 3)
        p[0::3] = centers[:, 0]
        p[1::3] = centers[:, 1]
        p[2::3] = 0.09
        inits.append(p)
        
    # 2. Random dense configurations
    for _ in range(10):
        centers = np.random.rand(N, 2) * 0.8 + 0.1
        p = np.zeros(N * 3)
        p[0::3] = centers[:, 0]
        p[1::3] = centers[:, 1]
        p[2::3] = 0.07
        inits.append(p)
        
    # 3. Perturbed 5x5 grid + 1 center circle
    for _ in range(5):
        gs = np.linspace(0.12, 0.88, 5)
        cx, cy = np.meshgrid(gs, gs)
        centers = np.column_stack((cx.flatten(), cy.flatten()))
        centers = np.vstack([centers, [0.5, 0.5]])
        centers += np.random.normal(0, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        p = np.zeros(N * 3)
        p[0::3] = centers[:, 0]
        p[1::3] = centers[:, 1]
        p[2::3] = 0.085
        inits.append(p)
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(123)
    best_p = None
    best_sum = -np.inf
    
    # Phase 1: Multi-start SLSQP optimization
    inits = generate_inits()
    for p0 in inits:
        p_opt, succ = run_slsqp(p0, maxiter=6000)
        c_vals = constraints(p_opt)
        if np.all(c_vals >= -1e-8):
            s = -objective(p_opt)
            if s > best_sum:
                best_sum = s
                best_p = p_opt.copy()
                
    # Phase 2: Perturbation & Local Search to escape local minima
    if best_p is not None:
        for _ in range(40):
            p_curr = best_p + np.random.normal(0, 0.0012, best_p.shape)
            p_curr[2::3] = np.clip(p_curr[2::3], 1e-6, 0.5)
            p_curr[0::3] = np.clip(p_curr[0::3], 0.01, 0.99)
            p_curr[1::3] = np.clip(p_curr[1::3], 0.01, 0.99)
            
            p_opt, succ = run_slsqp(p_curr, maxiter=4000)
            c_vals = constraints(p_opt)
            if np.all(c_vals >= -1e-8):
                s = -objective(p_opt)
                if s > best_sum:
                    best_sum = s
                    best_p = p_opt.copy()
                    
    # Phase 3: LP Repair & Center Adjustment
    # Fix centers, solve LP for maximal radii, then polish. This often finds tighter packings.
    if best_p is not None:
        for _ in range(15):
            centers = np.column_stack((best_p[0::3], best_p[1::3]))
            centers_pert = centers + np.random.normal(0, 0.0008, centers.shape)
            centers_pert = np.clip(centers_pert, 0.02, 0.98)
            
            radii_lp = solve_lp_radii(centers_pert)
            if radii_lp is not None:
                s_lp = np.sum(radii_lp)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_p = np.zeros(N*3)
                    best_p[0::3] = centers_pert[:, 0]
                    best_p[1::3] = centers_pert[:, 1]
                    best_p[2::3] = radii_lp
                    
                    p_opt, succ = run_slsqp(best_p, maxiter=5000)
                    c_vals = constraints(p_opt)
                    if np.all(c_vals >= -1e-8):
                        s_opt = -objective(p_opt)
                        if s_opt > best_sum:
                            best_sum = s_opt
                            best_p = p_opt.copy()
                            
    # Phase 4: Final Precise Polish
    if best_p is not None:
        p_opt, _ = run_slsqp(best_p, maxiter=8000)
        c_vals = constraints(p_opt)
        if np.all(c_vals >= -1e-8):
            s = -objective(p_opt)
            if s > best_sum:
                best_sum = s
                best_p = p_opt.copy()
                
    # Extract results
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3].copy()
    
    # Final safety validation and strict shrinking if necessary
    for _ in range(20):
        valid = True
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                valid = False
                break
        if not valid: break
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid: break
        if valid: break
        radii *= 0.99995
        
    return centers, np.maximum(radii, 0.0), float(np.sum(radii))

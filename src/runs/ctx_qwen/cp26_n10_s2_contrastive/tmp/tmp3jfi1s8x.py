import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure for efficiency
A_ub_lp = np.zeros((N_PAIRS, N))
A_ub_lp[np.arange(N_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(N_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-12, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def objective_joint(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + N_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    dists = np.hypot(dx, dy)
    c[4*N:] = dists - (r[I_IDX] + r[J_IDX])
    return c

def lp_objective_for_nm(center_flat):
    """Objective for Nelder-Mead: negative sum of radii from LP given centers."""
    centers = center_flat.reshape(N, 2)
    _, s = solve_lp_radii(centers)
    return -s

def generate_boundary_optimized_init(rng, pattern_type='hex'):
    """Generate initial configurations optimized for boundary exploitation."""
    centers = np.zeros((N, 2))
    idx = 0
    
    if pattern_type == 'hex_boundary':
        # Hexagonal lattice with tighter spacing near boundaries
        spacing = 0.16 + rng.uniform(-0.01, 0.02)
        margin = 0.04
        y = margin + spacing / 2
        row = 0
        while idx < N and y < 1.0 - margin:
            x_start = margin + (row % 2) * spacing / 2
            col = 0
            while x_start + col * spacing < 1.0 - margin and idx < N:
                centers[idx] = [x_start + col * spacing, y]
                idx += 1
                col += 1
            y += spacing * np.sqrt(3) / 2
            row += 1
    elif pattern_type == 'corner_heavy':
        # Place circles in corners and edges first, then fill center
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        edges = [[0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        for p in corners + edges:
            if idx < N:
                centers[idx] = p
                idx += 1
        # Fill remaining with hex pattern
        sp = 0.17
        y = 0.2
        row = 0
        while idx < N and y < 0.8:
            x = 0.2 + (row % 2) * sp / 2
            while x < 0.8 and idx < N:
                centers[idx] = [x, y]
                idx += 1
                x += sp
            y += sp * np.sqrt(3) / 2
            row += 1
    elif pattern_type == 'dense_grid':
        # Dense grid pattern
        step = 0.15
        y = step / 2
        while idx < N and y < 1.0 - step / 2:
            x = step / 2
            while x < 1.0 - step / 2 and idx < N:
                centers[idx] = [x, y]
                idx += 1
                x += step
            y += step
            
    # Fill remaining
    while idx < N:
        centers[idx] = rng.uniform(0.15, 0.85, 2)
        idx += 1
        
    # Add small noise
    centers += rng.normal(0, 0.004, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def optimize_joint_slsqp(c0, r0):
    """Run SLSQP joint optimization."""
    x0 = np.zeros(3 * N)
    x0[0::3] = c0[:, 0]
    x0[1::3] = c0[:, 1]
    x0[2::3] = np.maximum(r0, 1e-5)
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_joint}
    
    try:
        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        c_opt = res.x[:2*N].reshape(N, 2)
        r_opt, s_opt = solve_lp_radii(c_opt)
        return c_opt, r_opt, s_opt
    except Exception:
        return c0, r0, np.sum(r0)

def optimize_centers_nm(c0):
    """Optimize centers using Nelder-Mead with LP objective."""
    x0 = c0.flatten()
    try:
        res = minimize(lp_objective_for_nm, x0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
        c_opt = res.x.reshape(N, 2)
        r_opt, s_opt = solve_lp_radii(c_opt)
        return c_opt, r_opt, s_opt
    except Exception:
        _, s = solve_lp_radii(c0)
        return c0, np.full(N, 1e-6), s

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = 0.0
    best_centers = None
    best_radii = None
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse initial configurations with joint SLSQP
    inits = []
    for _ in range(15):
        inits.append(generate_boundary_optimized_init(rng, 'hex_boundary'))
    for _ in range(10):
        inits.append(generate_boundary_optimized_init(rng, 'corner_heavy'))
    for _ in range(10):
        inits.append(generate_boundary_optimized_init(rng, 'dense_grid'))
    for _ in range(20):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for c_init in inits:
        r_init, s_init = solve_lp_radii(c_init)
        c_opt, r_opt, s_opt = optimize_joint_slsqp(c_init, r_init)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
    # Phase 2: Nelder-Mead center optimization on best finds
    if best_centers is not None:
        for _ in range(8):
            c_nm, r_nm, s_nm = optimize_centers_nm(best_centers)
            if s_nm > best_sum:
                best_sum = s_nm
                best_centers = c_nm.copy()
                best_radii = r_nm.copy()
                # Polish with SLSQP after NM improvement
                c_pol, r_pol, s_pol = optimize_joint_slsqp(best_centers, best_radii)
                if s_pol > best_sum:
                    best_sum = s_pol
                    best_centers = c_pol.copy()
                    best_radii = r_pol.copy()
                    
    # Phase 3: Simulated Annealing Basin Hopping
    if best_centers is not None:
        cur_c = best_centers.copy()
        cur_r = best_radii.copy()
        cur_s = best_sum
        temp = 0.015
        
        for step in range(200):
            temp = 0.015 * np.exp(-step / 60.0)
            noise_scale = temp
            c_pert = cur_c + rng.normal(0, noise_scale, cur_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            # Accept if better, or probabilistically if worse
            if s_pert > cur_s or rng.random() < np.exp((s_pert - cur_s) / max(temp, 1e-4)):
                cur_c = c_pert
                cur_r = r_pert
                cur_s = s_pert
                
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_centers = cur_c.copy()
                    best_radii = cur_r.copy()
                    # Intensive local polish after new best
                    c_pol, r_pol, s_pol = optimize_joint_slsqp(best_centers, best_radii)
                    if s_pol > best_sum:
                        best_sum = s_pol
                        best_centers = c_pol.copy()
                        best_radii = r_pol.copy()
                        
    # Phase 4: Targeted subset perturbations
    if best_centers is not None:
        for trial in range(100):
            # Perturb 1-4 random circles
            k = rng.integers(1, 5)
            idxs = rng.choice(N, k, replace=False)
            c_pert = best_centers.copy()
            noise = rng.uniform(0.002, 0.01, (k, 2))
            c_pert[idxs] += noise
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                # Quick polish
                c_pol, r_pol, s_pol = optimize_joint_slsqp(best_centers, best_radii)
                if s_pol > best_sum:
                    best_sum = s_pol
                    best_centers = c_pol.copy()
                    best_radii = r_pol.copy()
                    
    # Fallback
    if best_centers is None:
        best_centers = generate_boundary_optimized_init(rng, 'hex_boundary')
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 5: Strict post-processing
    radii = best_radii.copy()
    
    # Enforce boundaries strictly
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-10)
        radii[i] = max(0.0, radii[i])
        
    # Resolve overlaps iteratively
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(best_centers[i, 0] - best_centers[j, 0], 
                               best_centers[i, 1] - best_centers[j, 1])
                if d < radii[i] + radii[j] - 1e-11:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))
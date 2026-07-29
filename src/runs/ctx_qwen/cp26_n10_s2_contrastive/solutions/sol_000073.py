# sol_000073 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000036 (state d4cf115e) state=7687f125 sum of radii=2.621304 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    m = len(I)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), I] = 1.0
    A_ub[np.arange(m), J] = 1.0
    
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, ub)))
        
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
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + len(I))
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    dist = np.hypot(dx, dy)
    c[4*N:] = dist - (r[I] + r[J])
    return c

def make_valid(centers, radii):
    """Project configuration to strictly satisfy boundary and overlap constraints."""
    for i in range(N):
        x, y = centers[i]
        max_r = min(x, 1.0-x, y, 1.0-y)
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r - 1e-9)
            
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-12:
                    excess = radii[i] + radii[j] - d
                    radii[i] -= excess/2.0
                    radii[j] -= excess/2.0
                    changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return centers, radii

def generate_init(seed, style='hex'):
    """Generate diverse initial center configurations."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    
    if style == 'hex':
        s = 0.16 + rng.uniform(-0.03, 0.05)
        idx = 0
        row = 0
        y = s/2 + rng.uniform(0, 0.05)
        while idx < N and y < 1.0 - s/2:
            x_start = s/2 + (row % 2) * s/2 + rng.uniform(0, 0.02)
            col = 0
            while x_start + col*s < 1.0 - s/2 and idx < N:
                centers[idx, 0] = x_start + col*s
                centers[idx, 1] = y
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            centers[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
    else:
        centers = rng.uniform(0.1, 0.9, (N, 2))
        
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing():
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Broad multi-start search with diverse lattice patterns
    for seed in range(60):
        style = 'hex' if seed < 45 else 'random'
        c0 = generate_init(seed, style)
        r0, _ = solve_radii_lp(c0)
        r0 = np.maximum(r0 * 0.95, 1e-4)  # Slight shrink ensures strict feasibility for SLSQP start
        
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            curr_centers = np.column_stack((cx, cy))
            
            # LP refinement extracts true maximal radii for optimized centers
            r_opt, s_opt = solve_radii_lp(curr_centers)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = curr_centers.copy()
                best_radii = r_opt.copy()
        except Exception:
            continue

    # Phase 2: Iterative Joint + LP refinement to push past local minima
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for _ in range(25):
            x0 = np.zeros(3*N)
            x0[0::3] = curr_c[:, 0]
            x0[1::3] = curr_c[:, 1]
            x0[2::3] = curr_r * 0.99
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                cx = res.x[0::3]
                cy = res.x[1::3]
                c_new = np.column_stack((cx, cy))
                
                r_new, s_new = solve_radii_lp(c_new)
                if s_new > curr_s:
                    curr_c = c_new
                    curr_r = r_new
                    curr_s = s_new
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
                    best_sum = curr_s
            except Exception:
                pass
                
            # Random perturbation to escape basins
            rng = np.random.RandomState(_)
            c_pert = curr_c + rng.normal(0, 0.002, curr_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert, s_pert = solve_radii_lp(c_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                curr_c = c_pert
                curr_r = r_pert
                
    # Phase 3: Simulated annealing style local search with decaying noise
    if best_centers is not None:
        for step in range(30):
            rng = np.random.RandomState(step * 17 + 42)
            noise_scale = 0.005 * np.exp(-step / 10.0)
            c_hop = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            c_hop = np.clip(c_hop, 0.02, 0.98)
            r_hop, s_hop = solve_radii_lp(c_hop)
            
            if s_hop > best_sum:
                best_sum = s_hop
                best_centers = c_hop.copy()
                best_radii = r_hop.copy()
                
                # Polish new best with SLSQP
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = best_radii * 0.98
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    c_ref = np.column_stack((cx, cy))
                    r_ref, s_ref = solve_radii_lp(c_ref)
                    if s_ref > best_sum:
                        best_sum = s_ref
                        best_centers = c_ref.copy()
                        best_radii = r_ref.copy()
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        best_centers = generate_init(0)
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # Strict post-processing to guarantee validity within numerical tolerance
    best_centers, best_radii = make_valid(best_centers, best_radii)
    return best_centers, best_radii, float(np.sum(best_radii))

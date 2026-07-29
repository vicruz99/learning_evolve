# sol_000076 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000036 (state d4cf115e) state=705d8cf5 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2*N:])

def constraints(x):
    """Compute inequality constraints (must be >= 0)."""
    cx = x[0:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    n_pairs = N*(N-1)//2
    c = np.empty(4*N + n_pairs)
    
    # Boundary constraints
    c[0:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    # Pairwise non-overlap constraints
    dx = cx[PAIRS_I] - cx[PAIRS_J]
    dy = cy[PAIRS_I] - cy[PAIRS_J]
    c[4*N:] = np.hypot(dx, dy) - r[PAIRS_I] - r[PAIRS_J]
    
    return c

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    n_pairs = n*(n-1)//2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    bounds = []
    for i in range(n):
        mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
        
    return np.full(n, 1e-7)

def run_single_opt(centers, radii, maxiter=5000):
    """Run SLSQP optimization from a given starting point."""
    x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False})
    
    if res.success or -res.fun > 0:
        c_opt = np.column_stack((res.x[0:N], res.x[N:2*N]))
        r_opt = np.maximum(res.x[2*N:], 1e-7)
        return c_opt, r_opt, -res.fun
    return centers, radii, np.sum(radii)

def generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # Hexagonal lattices with varying spacing
    for s in np.linspace(0.14, 0.19, 8):
        pts = []
        y = s/2
        row = 0
        while len(pts) < N:
            x = s/2 + (row % 2) * s/2
            while x < 1.0 - s/2 and len(pts) < N:
                pts.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        pts = np.array(pts[:N])
        
        for noise_scale in [0.0, 0.01, 0.03]:
            rng = np.random.RandomState(int(s*1000) + int(noise_scale*1000))
            p = pts + rng.normal(0, noise_scale, pts.shape)
            p = np.clip(p, 0.02, 0.98)
            r = solve_lp_radii(p) * 0.95
            inits.append((p, r))
            
    # Random uniform starts
    for seed in range(15):
        rng = np.random.RandomState(seed*13+7)
        p = rng.uniform(0.1, 0.9, (N, 2))
        r = solve_lp_radii(p) * 0.95
        inits.append((p, r))
        
    return inits

def run_packing():
    """Main packing routine: multi-start SLSQP + LP-guided local search."""
    best_sum = 0.0
    best_c = None
    best_r = None
    
    inits = generate_inits()
    
    # Phase 1: Broad SLSQP search from diverse starts
    for c0, r0 in inits:
        c, r, s = run_single_opt(c0, r0, maxiter=5000)
        # LP refinement guarantees optimal radii for these centers
        r_lp = solve_lp_radii(c)
        s_lp = np.sum(r_lp)
        if s_lp > best_sum:
            best_sum = s_lp
            best_c = c.copy()
            best_r = r_lp.copy()
            
    # Phase 2: LP-guided perturbation search (climbing the radius envelope)
    if best_c is not None:
        current_c = best_c.copy()
        current_r = best_r.copy()
        current_sum = best_sum
        
        sigma = 0.025
        for step in range(150):
            rng = np.random.RandomState(step * 31 + 11)
            c_pert = current_c + rng.normal(0, sigma, current_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            # Accept improvement
            if s_pert > current_sum:
                current_sum = s_pert
                current_c = c_pert
                current_r = r_pert
                
                # Occasional SLSQP polish to refine geometry
                if step % 10 == 0:
                    c_opt, r_opt, s_opt = run_single_opt(current_c, current_r * 0.95, maxiter=4000)
                    r_opt = solve_lp_radii(c_opt)
                    s_opt = np.sum(r_opt)
                    if s_opt > current_sum:
                        current_sum = s_opt
                        current_c = c_opt
                        current_r = r_opt
                        
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_c = current_c.copy()
                    best_r = current_r.copy()
                    
            # Cooling schedule
            sigma *= 0.96
            
    # Phase 3: Final rigorous SLSQP polish
    if best_c is not None:
        c, r, s = run_single_opt(best_c, best_r * 0.98, maxiter=10000)
        r_lp = solve_lp_radii(c)
        s_lp = np.sum(r_lp)
        if s_lp > best_sum:
            best_sum = s_lp
            best_c = c
            best_r = r_lp
            
    # Phase 4: Strict post-processing for numerical validity
    if best_c is not None:
        # Enforce boundary constraints strictly
        for i in range(N):
            mx = min(best_c[i,0], 1.0-best_c[i,0], best_c[i,1], 1.0-best_c[i,1])
            best_r[i] = min(best_r[i], mx - 1e-9)
            best_r[i] = max(best_r[i], 0.0)
            
        # Iteratively resolve any remaining overlaps
        for _ in range(100):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
                    if d < best_r[i] + best_r[j] - 1e-10:
                        exc = best_r[i] + best_r[j] - d
                        best_r[i] -= exc/2.0
                        best_r[j] -= exc/2.0
                        changed = True
            if not changed:
                break
                
        best_sum = float(np.sum(best_r))
        
    else:
        # Fallback (should not be reached)
        best_c = np.random.uniform(0.2, 0.8, (N, 2))
        best_r = solve_lp_radii(best_c)
        best_sum = float(np.sum(best_r))
        
    return best_c, best_r, best_sum

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute constant LP constraint matrix structure for speed
A_LP = np.zeros((N_PAIRS, N))
A_LP[np.arange(N_PAIRS), I_IDX] = 1.0
A_LP[np.arange(N_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r = np.maximum(res.x, 0.0)
            return r, np.sum(r)
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective_joint(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * N:])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    c = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    
    b = np.empty(4 * N + N_PAIRS)
    b[:N] = c[:, 0] - r
    b[N:2 * N] = 1.0 - c[:, 0] - r
    b[2 * N:3 * N] = c[:, 1] - r
    b[3 * N:4 * N] = 1.0 - c[:, 1] - r
    
    dx = c[I_IDX, 0] - c[J_IDX, 0]
    dy = c[I_IDX, 1] - c[J_IDX, 1]
    dists = np.hypot(dx, dy)
    b[4 * N:] = dists - (r[I_IDX] + r[J_IDX])
    return b

def optimize_joint(c0, r0, rng, iters=10000):
    """Run SLSQP to jointly optimize centers and radii from a starting point."""
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_joint}
    x0 = np.concatenate([c0.flatten(), np.maximum(r0, 1e-6)])
    
    try:
        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds,
                       constraints=cons,
                       options={'maxiter': iters, 'ftol': 1e-14, 'disp': False})
        if not np.isnan(res.fun):
            curr_sum = -res.fun
            init_sum = np.sum(r0)
            if res.success or curr_sum > init_sum:
                c_new = res.x[:2 * N].reshape(N, 2)
                # Re-solve LP to extract exact maximal radii for the optimized centers
                r_new, _ = solve_lp_radii(c_new)
                return c_new, r_new
    except Exception:
        pass
    return c0, r0

def run_simulated_annealing(c_init, r_init, rng, steps=4000):
    """Simulated annealing on circle centers, evaluating sum of radii via LP."""
    curr_c = c_init.copy()
    curr_r = r_init.copy()
    curr_s = np.sum(curr_r)
    
    best_c = curr_c.copy()
    best_r = curr_r.copy()
    best_s = curr_s
    
    temp = 0.015
    
    for step in range(steps):
        # Adaptive perturbation scale based on temperature
        noise_scale = 0.02 * np.sqrt(max(temp, 1e-4))
        
        # Perturb a random subset of circles to balance global/local search
        k = rng.integers(1, 6)
        idxs = rng.choice(N, k, replace=False)
        new_c = curr_c.copy()
        new_c[idxs] += rng.normal(0, noise_scale, (k, 2))
        new_c = np.clip(new_c, 0.01, 0.99)
        
        new_r, new_s = solve_lp_radii(new_c)
        if new_s > 0:
            delta = new_s - curr_s
            accept = delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-6))
            if accept:
                curr_c = new_c
                curr_r = new_r
                curr_s = new_s
                if curr_s > best_s:
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                    best_s = curr_s
                    
        temp *= 0.996
    return best_c, best_r, best_s

def generate_inits(rng):
    """Generate diverse initial configurations."""
    inits = []
    # Hexagonal lattices with varying spacing
    for sp in np.linspace(0.15, 0.22, 8):
        for seed in range(5):
            c = np.zeros((N, 2))
            idx = 0
            row = 0
            y = sp / 2
            while idx < N and y < 1.0 - sp / 2:
                x_start = sp / 2 + (row % 2) * sp / 2
                col = 0
                while x_start + col * sp < 1.0 - sp / 2 and idx < N:
                    c[idx] = [x_start + col * sp, y]
                    idx += 1
                    col += 1
                y += sp * np.sqrt(3) / 2
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            c += rng.normal(0, 0.005, c.shape)
            c = np.clip(c, 0.02, 0.98)
            r, _ = solve_lp_radii(c)
            inits.append((c, r))
            
    # Random feasible placements
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (N, 2))
        r, _ = solve_lp_radii(c)
        inits.append((c, r))
        
    return inits

def fix_violations(centers, radii):
    """Deterministically resolve overlaps and boundary violations."""
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], 
                 centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > mx:
            radii[i] = max(0.0, mx - 1e-9)
            
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = math.hypot(dx, dy)
                if d < radii[i] + radii[j] - 1e-12:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    return centers, np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_inits(rng)
    
    # Phase 1: Initial optimization from diverse starts
    for c0, r0 in inits:
        c_opt, r_opt = optimize_joint(c0, r0, rng, iters=8000)
        s_opt = np.sum(r_opt)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
    # Phase 2: Simulated Annealing to escape local minima
    if best_centers is not None:
        c_sa, r_sa, s_sa = run_simulated_annealing(best_centers, best_radii, rng, steps=4000)
        if s_sa > best_sum:
            best_sum = s_sa
            best_centers = c_sa.copy()
            best_radii = r_sa.copy()
            
        # Polish SA result with joint optimization
        c_pol, r_pol = optimize_joint(best_centers, best_radii, rng, iters=12000)
        s_pol = np.sum(r_pol)
        if s_pol > best_sum:
            best_sum = s_pol
            best_centers = c_pol.copy()
            best_radii = r_pol.copy()
            
    # Phase 3: Targeted local jumps for fine-tuning
    if best_centers is not None:
        for _ in range(50):
            c_pert = best_centers.copy()
            k = rng.integers(2, 6)
            idxs = rng.choice(N, k, replace=False)
            c_pert[idxs] += rng.normal(0, 0.005, (k, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                c_pol, r_pol = optimize_joint(best_centers, best_radii, rng, iters=6000)
                if np.sum(r_pol) > best_sum:
                    best_sum = np.sum(r_pol)
                    best_centers = c_pol.copy()
                    best_radii = r_pol.copy()
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = rng.uniform(0.2, 0.8, (N, 2))
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Final strict post-processing to guarantee validator compliance
    best_centers, best_radii = fix_violations(best_centers, best_radii)
    return best_centers, best_radii, float(np.sum(best_radii))
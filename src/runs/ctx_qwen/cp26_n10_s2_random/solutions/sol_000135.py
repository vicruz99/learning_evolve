# sol_000135 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000124 (state e4120b9c) state=032311e9 sum of radii=2.627681 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IDX_I, IDX_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(IDX_I)

def get_lp_matrix():
    """Precomputes the inequality constraint matrix for the radii LP."""
    n = N
    num_bound = n
    A = np.zeros((NUM_PAIRS + num_bound, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            idx += 1
    for i in range(n):
        A[idx + i, i] = 1.0
    return A

A_LP = get_lp_matrix()

def solve_lp_radii(centers):
    """Solves LP to find maximum sum of radii for fixed centers."""
    n = centers.shape[0]
    c = np.clip(centers, 1e-6, 1.0 - 1e-6)
    
    # Upper bounds from boundaries: r_i <= min(x, 1-x, y, 1-y)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 0.0)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Construct b_ub
    b = np.empty(NUM_PAIRS + n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            b[idx] = dists[i, j]
            idx += 1
    b[idx:] = ub
    
    # Solve LP: max sum(r) s.t. A_ub r <= b, 0 <= r <= ub
    bounds_r = [(0.0, u) for u in ub]
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, bounds=bounds_r, method='highs')
    
    if res.success:
        return -res.fun, res.x
    return 0.0, np.zeros(n)

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_joint(v):
    """Computes boundary and non-overlap constraints (must be >= 0). Uses squared distances."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    
    cons = []
    # Boundary constraints
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints (squared to avoid sqrt gradient issues)
    dx = c[IDX_I, 0] - c[IDX_J, 0]
    dy = c[IDX_I, 1] - c[IDX_J, 1]
    dr = r[IDX_I] + r[IDX_J]
    cons.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(cons)

def generate_starts(rng):
    """Generates diverse initial center configurations."""
    starts = []
    
    # 1. Hexagonal lattice patterns with various row structures
    patterns = [[5,5,5,5,6], [5,6,5,6,4], [6,5,6,5,4], [4,6,6,6,4], [5,5,6,5,5], [6,4,6,5,5]]
    for pat in patterns:
        pts = []
        y = 0.09
        r_est = 0.09
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                if len(pts) < N:
                    pts.append([x, y])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3)
        starts.append(np.array(pts[:N]))
        
    # 2. Force-repulsion spread (simulates dense packing)
    for _ in range(6):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            f_mag = 0.01 / (dists**2 + 0.0001)
            forces += np.sum(diff * f_mag[:, :, None], axis=1)
            c += 0.005 * forces
            c = np.clip(c, 0.02, 0.98)
        starts.append(c)
        
    # 3. Random dense starts
    for _ in range(15):
        starts.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return starts

def run_packing():
    rng = np.random.default_rng(42)
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': cons_joint}
    
    starts = generate_starts(rng)
    
    # --- Phase 1: Multi-start SLSQP Joint Optimization ---
    for c_init in starts:
        c_init = np.clip(c_init, 0.02, 0.98)
        _, r_init = solve_lp_radii(c_init)
        # Shrink radii slightly to ensure strict initial feasibility for SLSQP
        r_init = np.maximum(r_init * 0.92, 0.005)
        v0 = np.concatenate([c_init.flatten(), r_init])
        
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                           constraints=cons_dict, options={'maxiter': 15000, 'ftol': 1e-13, 'catol': 1e-13})
            
            # Check strict feasibility
            if np.min(cons_joint(res.x)) >= -1e-8:
                c_opt = res.x[:2 * N].reshape(N, 2)
                s_lp, r_lp = solve_lp_radii(c_opt)
                
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # --- Phase 2: Greedy Basin Hopping on Centers (LP Objective) ---
    if best_centers is not None:
        curr_c = best_centers.copy()
        
        # Run multiple passes with decaying noise
        for pass_idx in range(3):
            noise_scale = 0.006 * (0.7 ** pass_idx)
            for step in range(80):
                c_trial = curr_c + rng.normal(0, noise_scale, curr_c.shape)
                c_trial = np.clip(c_trial, 0.01, 0.99)
                
                s_trial, _ = solve_lp_radii(c_trial)
                
                # Greedy accept
                if s_trial > best_sum + 1e-9:
                    best_sum = s_trial
                    best_centers = c_trial.copy()
                    best_radii, _ = solve_lp_radii(best_centers) # Update radii
                    curr_c = c_trial.copy()
                # Occasionally accept neutral moves to escape plateaus
                elif rng.random() < 0.05:
                    curr_c = c_trial.copy()
        
        # Final SLSQP polish on the best geometry found
        r_init = best_radii * 0.94
        v0 = np.concatenate([best_centers.flatten(), r_init])
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                           constraints=cons_dict, options={'maxiter': 20000, 'ftol': 1e-13})
            if np.min(cons_joint(res.x)) >= -1e-8:
                c_opt = res.x[:2 * N].reshape(N, 2)
                s_lp, r_lp = solve_lp_radii(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # --- Phase 3: Strict Numerical Repair ---
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    for _ in range(80):
        changed = False
        
        # Resolve overlaps by symmetric shrinking
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Clamp to boundaries
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-12:
                radii[i] = mr
                changed = True
                
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))

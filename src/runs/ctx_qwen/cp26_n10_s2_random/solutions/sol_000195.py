# sol_000195 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000163 (state a7643fac) state=5daa0ffe sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
PAIR_INDICES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_PAIRS = len(PAIR_INDICES)

# Precompute constant structure of the LP constraint matrix
A_ub_structure = np.zeros((N_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(PAIR_INDICES):
    A_ub_structure[k, i] = 1.0
    A_ub_structure[k, j] = 1.0
for i in range(N):
    base = N_PAIRS + 4 * i
    A_ub_structure[base, i] = 1.0
    A_ub_structure[base + 1, i] = 1.0
    A_ub_structure[base + 2, i] = 1.0
    A_ub_structure[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient via duals."""
    c = np.clip(centers, 1e-7, 1.0 - 1e-7)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(N_PAIRS + 4 * N)
    for k, (i, j) in enumerate(PAIR_INDICES):
        b_ub[k] = dists[i, j]
    for i in range(N):
        base = N_PAIRS + 4 * i
        b_ub[base] = c[i, 0]
        b_ub[base + 1] = 1.0 - c[i, 0]
        b_ub[base + 2] = c[i, 1]
        b_ub[base + 3] = 1.0 - c[i, 1]
        
    bounds = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_ub_structure, b_ub=b_ub, 
                  bounds=bounds, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract dual marginals safely across scipy versions
    duals = np.zeros(b_ub.shape[0])
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(c)
    
    # Pairwise repulsion forces from active distance constraints
    for k, (i, j) in enumerate(PAIR_INDICES):
        mu = duals[k]
        if mu > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
                
    # Boundary forces from active wall constraints
    for i in range(N):
        base = N_PAIRS + 4 * i
        mu_L = duals[base]
        mu_R = duals[base + 1]
        mu_B = duals[base + 2]
        mu_T = duals[base + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, s_sum, grad

def optimize_gradient(centers0, steps=2000, init_step=0.015):
    """Runs gradient ascent on centers with backtracking line search."""
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    
    radii, curr_sum, grad = solve_lp_and_grad(centers)
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_centers = centers.copy()
        
    for k in range(steps):
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-11:
            break
            
        direction = grad / grad_norm
        
        # Backtracking line search
        improved = False
        trial_step = step
        for _ in range(12):
            trial_c = centers + trial_step * direction
            trial_c = np.clip(trial_c, 1e-5, 1.0 - 1e-5)
            _, trial_sum, _ = solve_lp_and_grad(trial_c)
            
            if trial_sum > curr_sum + 1e-13:
                centers = trial_c
                curr_sum = trial_sum
                _, _, grad = solve_lp_and_grad(centers)
                improved = True
                break
            trial_step *= 0.4
            
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break
        else:
            step = min(step * 1.1, 0.03)
            
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            
        # Occasional jitter to escape flat gradients
        if k > 0 and k % 300 == 0:
            centers += np.random.normal(0, 0.0008, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            radii, curr_sum, grad = solve_lp_and_grad(centers)
            
    return best_centers, best_sum

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Computes boundary and non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
    idx_i, idx_j = np.triu_indices(N, 1)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    dr = r[idx_i] + r[idx_j]
    cons.append(np.sqrt(dx**2 + dy**2) - dr)
    return np.concatenate(cons)

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [7, 6, 6, 7], [6, 7, 6, 7], [7, 6, 7, 6]
    ]
    
    for pat in patterns:
        for r_est in [0.088, 0.095, 0.102, 0.109]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.0025, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # Random dense starts
    for _ in range(10):
        starts.append(rng.uniform(0.12, 0.88, (N, 2)))
        
    # Force-directed spread starts
    for _ in range(5):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.15 and dist > 1e-6:
                        push = (0.15 - dist) * 0.1
                        forces[i] += d_vec / dist * push
                        forces[j] -= d_vec / dist * push
            c += forces
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # --- Phase 1: Gradient ascent from diverse starts ---
    for i, c0 in enumerate(starts):
        c_opt, s_opt = optimize_gradient(c0, steps=2500, init_step=0.012)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        best_r, _, _ = solve_lp_and_grad(best_c)
    else:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # --- Phase 2: Structured Perturbation & Restart ---
    # Decaying perturbation schedule to explore and refine
    pert_scale = 0.008
    for _ in range(12):
        c_pert = best_c + rng.normal(0, pert_scale, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        c_opt, s_opt = optimize_gradient(c_pert, steps=1800, init_step=0.009)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
        pert_scale *= 0.85
        
    # --- Phase 3: SLSQP Joint Polish ---
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_j = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    for _ in range(5):
        v_pert = v0 + rng.normal(0, 0.0012, v0.shape)
        v_pert = np.clip(v_pert, 0.01, 0.99)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.01, 0.4)
        try:
            res = minimize(obj_joint, v_pert, method='SLSQP', bounds=bounds_j,
                          constraints={'type': 'ineq', 'fun': cons_joint},
                          options={'maxiter': 6000, 'ftol': 1e-13})
            if np.min(cons_joint(res.x)) >= -1e-9:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass
            
    # --- Phase 4: Targeted Tight-Cluster Perturbation ---
    # Identify closest pairs and perturb them specifically to break symmetry traps
    for _ in range(8):
        c_try = best_c.copy()
        # Find 3 closest pairs
        dists = np.linalg.norm(c_try[:, np.newaxis, :] - c_try[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        flat_dists = dists[np.triu_indices(N, 1)]
        idx_pairs = np.argpartition(flat_dists, 3)[:3]
        
        for idx in idx_pairs:
            # Recover i, j from flat index
            k = 0
            found_i, found_j = -1, -1
            for i in range(N):
                for j in range(i+1, N):
                    if k == idx:
                        found_i, found_j = i, j
                    k += 1
            if found_i >= 0:
                c_try[found_i] += rng.normal(0, 0.004, 2)
                c_try[found_j] += rng.normal(0, 0.004, 2)
                
        c_try = np.clip(c_try, 0.02, 0.98)
        c_opt, s_opt = optimize_gradient(c_try, steps=1500, init_step=0.006)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)

    # --- Phase 5: Final SLSQP Polish ---
    r_lp, _, _ = solve_lp_and_grad(best_c)
    v0_final = np.concatenate([best_c.flatten(), r_lp])
    try:
        res = minimize(obj_joint, v0_final, method='SLSQP', bounds=bounds_j,
                      constraints={'type': 'ineq', 'fun': cons_joint},
                      options={'maxiter': 5000, 'ftol': 1e-13})
        if np.min(cons_joint(res.x)) >= -1e-9:
            s = np.sum(res.x[2*N:])
            if s > best_sum:
                best_sum = s
                best_c = res.x[:2*N].reshape(N, 2).copy()
                best_r = res.x[2*N:].copy()
    except Exception:
        pass
        
    # --- Phase 6: Strict Numerical Repair ---
    centers = best_c.copy()
    radii = best_r.copy()
    for _ in range(200):
        changed = False
        # Fix pairwise overlaps proportionally
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    overlap = req - d
                    total = radii[i] + radii[j]
                    if total > 1e-12:
                        fi = radii[i] / total
                        fj = radii[j] / total
                    else:
                        fi = fj = 0.5
                    shrink = (overlap + 1e-9) / 2.0
                    radii[i] -= shrink * fi
                    radii[j] -= shrink * fj
                    changed = True
        # Fix boundary violations
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))

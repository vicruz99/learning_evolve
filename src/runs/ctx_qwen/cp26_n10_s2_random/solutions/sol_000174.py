# sol_000174 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000147 (state da2cd853) state=0d101e7a sum of radii=2.613222 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
IDX_I, IDX_J = np.triu_indices(N, k=1)
N_PAIRS = len(IDX_I)

# Precompute constant LP constraint matrix structure
A_Ub_const = np.zeros((N_PAIRS, N))
A_Ub_const[np.arange(N_PAIRS), IDX_I] = 1.0
A_Ub_const[np.arange(N_PAIRS), IDX_J] = 1.0

def compute_lp_and_grad(centers):
    """Solves LP for radii and computes gradient of sum of radii w.r.t centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = dists[IDX_I, IDX_J]
    
    # Boundary limits for radii
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    bounds = [(0, u) for u in ub]
    
    res = linprog(-np.ones(N), A_ub=A_Ub_const, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        return None, None, None, None
        
    radii = res.x
    duals = res.ineqlin.marginals
    
    # Gradient computation from dual variables
    grad = np.zeros_like(centers)
    for k in range(N_PAIRS):
        lam = duals[k]
        if lam > 1e-9:
            i, j = IDX_I[k], IDX_J[k]
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
                
    return radii, -res.fun, grad, duals

def obj_func(x):
    """Objective for center optimization: minimize negative LP sum of radii."""
    centers = x.reshape(N, 2)
    _, val, _, _ = compute_lp_and_grad(centers)
    return -val if val is not None else 0.0

def jac_func(x):
    """Gradient for center optimization."""
    centers = x.reshape(N, 2)
    _, _, grad, _ = compute_lp_and_grad(centers)
    return -grad.flatten() if grad is not None else np.zeros_like(x)

def generate_hex_start(r0, shift=0.0, rng=None):
    """Generates a hexagonal lattice configuration."""
    if rng is None:
        rng = np.random.default_rng(42)
    centers = []
    y = r0
    while len(centers) < N:
        x = r0 + shift
        while x <= 1.0 - r0 and len(centers) < N:
            centers.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        shift = r0 if shift == 0.0 else 0.0
        
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.003, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint optimization: boundary and non-overlap."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    d = np.linalg.norm(c[IDX_I] - c[IDX_J], axis=1)
    con.append(d - (r[IDX_I] + r[IDX_J]))
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(123)
    bounds_c = [(0.01, 0.99)] * (2 * N)
    
    best_centers = None
    best_sum = -np.inf
    
    # --- Phase 1: Multi-Start L-BFGS-B with Exact Gradients ---
    starts = []
    # Diverse hexagonal lattices
    for r0 in np.arange(0.06, 0.125, 0.008):
        for shift in [0.0, r0]:
            starts.append(generate_hex_start(r0, shift, rng))
    # Random dense starts
    for _ in range(15):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    for c0 in starts:
        try:
            res = minimize(obj_func, c0.flatten(), jac=jac_func, method='L-BFGS-B',
                           bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-14})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = starts[0]
        
    # --- Phase 2: Perturbation & Basin Hopping ---
    for step in range(80):
        noise_scale = 0.004 * (0.96 ** step)
        c_trial = best_centers + rng.normal(0, noise_scale, best_centers.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        try:
            res = minimize(obj_func, c_trial.flatten(), jac=jac_func, method='L-BFGS-B',
                           bounds=bounds_c, options={'maxiter': 600, 'ftol': 1e-14})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    # --- Phase 3: Joint SLSQP Polish ---
    radii_lp, _, _, _ = compute_lp_and_grad(best_centers)
    if radii_lp is None:
        radii_lp = np.full(N, 0.08)
        
    v0 = np.concatenate([best_centers.flatten(), radii_lp])
    bounds_j = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res_j = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_j,
                         constraints={'type': 'ineq', 'fun': cons_joint},
                         options={'maxiter': 8000, 'ftol': 1e-14})
        if np.sum(res_j.x[2*N:]) > best_sum - 1e-8:
            best_centers = res_j.x[:2*N].reshape(N, 2)
            radii_lp = res_j.x[2*N:]
            best_sum = np.sum(radii_lp)
    except Exception:
        pass
        
    # --- Phase 4: Strict Numerical Repair ---
    centers = best_centers.copy()
    radii = radii_lp.copy()
    
    for _ in range(150):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-11
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Fix boundary violations
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
                
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum

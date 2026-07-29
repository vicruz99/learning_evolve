# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000063 (state 0dfa75ae) state=f2a6fc21 sum of radii=2.369685 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

# Problem dimensions
N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
N_PAIRS = len(PAIR_I)
N_BOUND = 4 * N

# Precompute LP constraint matrix A for: A @ r <= b_ub
# Shape: (N_PAIRS + N_BOUND, N)
A = np.zeros((N_PAIRS + N_BOUND, N))
for k in range(N_PAIRS):
    A[k, PAIR_I[k]] = 1.0
    A[k, PAIR_J[k]] = 1.0

for i in range(N):
    # r_i <= x_i
    A[N_PAIRS + 4*i, i] = 1.0
    # r_i <= 1 - x_i
    A[N_PAIRS + 4*i + 1, i] = 1.0
    # r_i <= y_i
    A[N_PAIRS + 4*i + 2, i] = 1.0
    # r_i <= 1 - y_i
    A[N_PAIRS + 4*i + 3, i] = 1.0

# LP objective: minimize -sum(r_i) => maximize sum(r_i)
c_obj = -np.ones(N)
bounds_r = [(0.0, 0.5)] * N

def compute_obj_and_grad(centers_flat):
    """
    Solves LP for optimal radii given centers.
    Returns (sum_radii, gradient_of_sum_radii_wrt_centers_flat).
    Uses LP dual variables for exact analytical gradients.
    """
    centers = centers_flat.reshape(N, 2)
    b_ub = np.zeros(N_PAIRS + N_BOUND)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub[:N_PAIRS] = dists[PAIR_I, PAIR_J]
    
    # Boundary distances
    x = centers[:, 0]
    y = centers[:, 1]
    b_ub[N_PAIRS:N_PAIRS+N] = x
    b_ub[N_PAIRS+N:N_PAIRS+2*N] = 1.0 - x
    b_ub[N_PAIRS+2*N:N_PAIRS+3*N] = y
    b_ub[N_PAIRS+3*N:N_PAIRS+4*N] = 1.0 - y
    
    # Solve LP
    res = linprog(c_obj, A_ub=A, b_ub=b_ub, bounds=bounds_r, method='highs')
    
    if not res.success:
        return 0.0, np.zeros_like(centers_flat)
        
    obj_val = -res.fun
    # Dual variables (marginals) correspond to the constraints A_ub @ r <= b_ub
    marginals = np.maximum(res.ineqlin.marginals, 0.0)
    
    # Compute gradient w.r.t centers using chain rule: grad = y^T * d(b)/d(centers)
    grad = np.zeros((N, 2))
    
    # Pairwise contributions: constraint r_i + r_j <= ||c_i - c_j||
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            mu = marginals[k]
            if mu > 1e-9:
                d = dists[i, j]
                if d > 1e-9:
                    vec = (centers[i] - centers[j]) / d
                    grad[i] += mu * vec
                    grad[j] -= mu * vec
            k += 1
            
    # Boundary contributions
    for i in range(N):
        # d(x_i)/dx_i = 1
        grad[i, 0] += marginals[N_PAIRS + 4*i]
        # d(1-x_i)/dx_i = -1
        grad[i, 0] -= marginals[N_PAIRS + 4*i + 1]
        # d(y_i)/dy_i = 1
        grad[i, 1] += marginals[N_PAIRS + 4*i + 2]
        # d(1-y_i)/dy_i = -1
        grad[i, 1] -= marginals[N_PAIRS + 4*i + 3]
        
    return obj_val, grad.flatten()

def obj_and_jac(centers_flat):
    """Wrapper for scipy minimize (expects minimization of -obj)"""
    val, grad = compute_obj_and_grad(centers_flat)
    return -val, -grad

def get_optimal_radii(centers):
    """Solves LP and returns optimal radii array"""
    centers = np.asarray(centers).reshape(N, 2)
    b_ub = np.zeros(N_PAIRS + N_BOUND)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub[:N_PAIRS] = dists[PAIR_I, PAIR_J]
    x = centers[:, 0]
    y = centers[:, 1]
    b_ub[N_PAIRS:N_PAIRS+N] = x
    b_ub[N_PAIRS+N:N_PAIRS+2*N] = 1.0 - x
    b_ub[N_PAIRS+2*N:N_PAIRS+3*N] = y
    b_ub[N_PAIRS+3*N:N_PAIRS+4*N] = 1.0 - y
    
    res = linprog(c_obj, A_ub=A, b_ub=b_ub, bounds=bounds_r, method='highs')
    if res.success:
        return res.x
    return np.zeros(N)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    best_val = -np.inf
    best_centers = None
    
    inits = []
    
    # 1. Hexagonal lattice initializations (various densities & shifts)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        centers = np.zeros((N, 2))
        r_init = 0.092 + rng.uniform(-0.008, 0.008)
        y = r_init
        row = 0
        idx = 0
        while idx < N:
            shift = (row % 2) * r_init
            x = r_init + shift
            while x + r_init <= 1.0 and idx < N:
                centers[idx] = [x, y]
                x += 2 * r_init
                idx += 1
            y += r_init * np.sqrt(3.0)
            row += 1
        centers += rng.uniform(-0.004, 0.004, (N, 2))
        centers = np.clip(centers, 1e-2, 1.0 - 1e-2)
        inits.append(centers.flatten())
        
    # 2. Dense random initializations
    for seed in range(15):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.12, 0.88, (N, 2))
        inits.append(c.flatten())
        
    # Phase 1: Multi-start local optimization
    for c0 in inits:
        try:
            res = minimize(obj_and_jac, c0, method='L-BFGS-B', 
                           bounds=bounds_c, jac=True,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-11})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_centers = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative perturbation to escape local minima
    for step in range(30):
        pert = best_centers.copy()
        pert += np.random.normal(0, 0.0015, pert.shape)
        pert = np.clip(pert, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_and_jac, pert, method='L-BFGS-B', 
                           bounds=bounds_c, jac=True,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-11})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_centers = res.x.copy()
        except Exception:
            pass

    # Extract final valid radii
    centers = best_centers.reshape(N, 2)
    radii = get_optimal_radii(centers)
    
    # Phase 3: Strict validation repair
    # Shrink radii slightly to guarantee pass within 1e-12 tolerance
    radii *= (1.0 - 1e-7)
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    
    # Ensure no boundary violations
    for i in range(N):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r
            
    # Resolve any residual overlaps by proportional shrinking
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d + 1e-12) / 2.0
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii

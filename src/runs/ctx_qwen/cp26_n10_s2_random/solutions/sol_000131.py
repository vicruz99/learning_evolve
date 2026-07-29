# sol_000131 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000104 (state 8c305e91) state=f5edcace sum of radii=2.310881 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

N = 26

def compute_lp_and_grad(centers):
    """
    Solves the LP to find optimal radii for fixed centers and computes the gradient
    of the sum of radii with respect to the centers using LP dual variables.
    """
    n = centers.shape[0]
    num_pairs = n * (n - 1) // 2
    num_bounds = 4 * n
    num_con = num_pairs + num_bounds
    
    # Build constraint matrix A_ub for r_i + r_j <= d_ij and r_i <= boundaries
    A_ub = np.zeros((num_con, n))
    idx = 0
    pair_ij = np.zeros((num_pairs, 2), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pair_ij[idx] = (i, j)
            idx += 1
            
    for i in range(n):
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        
    # Build RHS vector b_ub dependent on centers
    b_ub = np.zeros(num_con)
    idx = 0
    
    dx = centers[:, None, 0] - centers[None, :, 0]
    dy = centers[:, None, 1] - centers[None, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    for i in range(n):
        for j in range(i + 1, n):
            b_ub[idx] = dists[i, j]
            idx += 1
            
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    # Solve LP: maximize sum(r) <=> minimize -sum(r)
    c_obj = -np.ones(n)
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, method='highs')
    except Exception:
        return None, 0.0, np.zeros((n, 2))
        
    if not res.success:
        return None, 0.0, np.zeros((n, 2))
        
    radii = res.x
    # Extract dual variables (shadow prices)
    try:
        duals = res.ineqlin.marginals
    except AttributeError:
        duals = np.zeros(num_con)
    
    # Compute gradient using Envelope Theorem
    grad = np.zeros((n, 2))
    idx = 0
    
    # Pairwise contact forces
    for k in range(num_pairs):
        lam = duals[k]
        if lam > 1e-8:
            i, j = pair_ij[k]
            d = dists[i, j]
            if d > 1e-8:
                vec = centers[i] - centers[j]
                f = lam / d
                grad[i] += f * vec
                grad[j] -= f * vec
        idx += 1
        
    # Boundary contact forces
    for i in range(n):
        mu_L = duals[idx]; idx += 1  # r_i <= x_i
        mu_R = duals[idx]; idx += 1  # r_i <= 1-x_i
        mu_B = duals[idx]; idx += 1  # r_i <= y_i
        mu_T = duals[idx]; idx += 1  # r_i <= 1-y_i
        
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    sum_r = np.sum(radii)
    return radii, sum_r, grad

def objective(x):
    """Objective function for minimization: minimize negative sum of radii."""
    c = x.reshape(N, 2)
    _, s, _ = compute_lp_and_grad(c)
    return -s

def gradient(x):
    """Gradient of the objective function."""
    c = x.reshape(N, 2)
    _, _, g = compute_lp_and_grad(c)
    return -g.flatten()

def generate_inits():
    """Generates diverse initial configurations for multi-start optimization."""
    inits = []
    
    # Hexagonal lattice patterns
    patterns = [
        [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], 
        [5, 5, 6, 5, 5], [5, 5, 5, 6, 5], [5, 5, 5, 5, 5, 1],
        [6, 5, 5, 5, 4], [4, 6, 5, 5, 6], [5, 5, 6, 6, 4],
        [6, 6, 5, 5, 4], [5, 7, 5, 5, 4], [4, 5, 6, 5, 6]
    ]
    
    for pat in patterns:
        if sum(pat) != N:
            continue
        centers = []
        r_est = 0.102
        dy = np.sqrt(3) * r_est
        for r_idx, count in enumerate(pat):
            y = r_est + r_idx * dy
            offset = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + offset
            for _ in range(count):
                centers.append([x, y])
                x += 2 * r_est
        inits.append(np.array(centers[:N]))
        
    # Dense random starts
    for _ in range(30):
        c = np.random.rand(N, 2) * 0.8 + 0.1
        inits.append(c)
        
    return inits

def repair(centers, radii):
    """Iteratively shrinks radii to strictly resolve overlaps and clamp to boundaries."""
    radii = radii.copy()
    for _ in range(30):
        changed = False
        for k in range(N):
            mx = min(centers[k, 0], 1.0 - centers[k, 0], 
                     centers[k, 1], 1.0 - centers[k, 1])
            if radii[k] > mx - 1e-10:
                radii[k] = mx
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(centers[i, 0] - centers[j, 0], 
                               centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    ov = radii[i] + radii[j] - d
                    radii[i] -= ov * 0.5
                    radii[j] -= ov * 0.5
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    np.random.seed(42)
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    bounds = [(0.02, 0.98)] * (2 * N)
    inits = generate_inits()
    
    for c0 in inits:
        # Add slight perturbation to escape symmetries
        c0 = c0 + np.random.normal(0, 0.003, c0.shape)
        c0 = np.clip(c0, 0.02, 0.98)
        
        try:
            res = opt.minimize(objective, c0.flatten(), method='L-BFGS-B', 
                               jac=gradient, bounds=bounds,
                               options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-9})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x.reshape(N, 2)
                best_radii, _, _ = compute_lp_and_grad(best_centers)
        except Exception:
            continue
            
    if best_radii is not None:
        radii = repair(best_centers, best_radii)
        # Final safety shrink to guarantee validator tolerance compliance
        radii *= 0.9999999
        radii = np.maximum(radii, 0.0)
        return best_centers, radii, float(np.sum(radii))
    else:
        # Fallback (should not trigger with this robust setup)
        c = np.random.rand(N, 2) * 0.8 + 0.1
        r = np.full(N, 0.01)
        return c, r, float(np.sum(r))

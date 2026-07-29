# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000078 (state b6b33190) state=9493557a sum of radii=2.516894 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def build_A_ub():
    """Precomputes the inequality constraint matrix for the LP."""
    M = N * (N - 1) // 2 + 4 * N
    A = np.zeros((M, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            idx += 1
    for i in range(N):
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
        A[idx, i] = 1.0; idx += 1
    return A

A_UB = build_A_ub()
NUM_PAIRS = N * (N - 1) // 2

def solve_lp_and_grad(centers):
    """
    Solves the LP to find optimal radii for fixed centers and computes the 
    gradient of the sum of radii with respect to centers using dual variables.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            b[idx] = dists[i, j]
            idx += 1
    for i in range(N):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(N), A_ub=A_UB, b_ub=b, bounds=(0, None), method='highs')
    
    if not res.success:
        return None, 0.0, np.zeros_like(centers)
        
    radii = res.x
    # Marginals are shadow prices; negative of them gives positive duals for active constraints
    duals = -res.ineqlin.marginals
    
    grad = np.zeros_like(centers)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            lam = duals[idx]
            if lam > 1e-9:
                d = dists[i, j]
                if d > 1e-9:
                    vec = (centers[i] - centers[j]) / d
                    grad[i] += lam * vec
                    grad[j] -= lam * vec
            idx += 1
            
    b_start = NUM_PAIRS
    for i in range(N):
        mu_x = duals[b_start + 4*i]
        mu_1x = duals[b_start + 4*i + 1]
        mu_y = duals[b_start + 4*i + 2]
        mu_1y = duals[b_start + 4*i + 3]
        grad[i, 0] += mu_x - mu_1x
        grad[i, 1] += mu_y - mu_1y
        
    return radii, np.sum(radii), grad

def get_hex_init(r0):
    """Generates a hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N:
        x = r0 if row % 2 == 0 else 2 * r0
        while x + r0 <= 1.0 and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    return np.array(pts[:N])

def obj_powell(c_flat):
    """Objective function for Powell optimization: maximize sum of radii."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, 1e-7, 1.0 - 1e-7)
    _, s, _ = solve_lp_and_grad(c)
    return -s

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_centers = None
    best_sum = -np.inf
    
    # Phase 1: Multi-start gradient ascent on centers
    for r0 in [0.08, 0.085, 0.09, 0.095, 0.10, 0.105]:
        centers = get_hex_init(r0)
        centers += np.random.normal(0, 0.003, centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        
        curr_centers = centers.copy()
        step = 0.003
        for it in range(800):
            _, curr_sum, grad = solve_lp_and_grad(curr_centers)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = curr_centers.copy()
                
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-9:
                curr_centers += step * grad / g_norm
            curr_centers += np.random.normal(0, 0.0008, curr_centers.shape)
            curr_centers = np.clip(curr_centers, 1e-5, 1.0 - 1e-5)
            step *= 0.999
            
    # Phase 2: Powell polish to escape local minima and fine-tune
    res = minimize(obj_powell, best_centers.flatten(), method='Powell',
                   bounds=[(0.0, 1.0)] * (2 * N),
                   options={'maxiter': 5000, 'ftol': 1e-14, 'xtol': 1e-14})
    if -res.fun > best_sum:
        best_sum = -res.fun
        best_centers = res.x.reshape(N, 2)
        
    # Phase 3: Exact LP for final radii
    radii, best_sum, _ = solve_lp_and_grad(best_centers)
    
    # Phase 4: Strict numerical repair to guarantee validator compliance
    for _ in range(50):
        changed = False
        for i in range(N):
            x, y = best_centers[i]
            mr = min(x, 1.0 - x, y, 1.0 - y)
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                             best_centers[i, 1] - best_centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))

# sol_000053 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000026 (state fcd5fdc4) state=25cf251f sum of radii=1.870996 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import basinhopping, minimize
import math

N_CIRCLES = 26

def compute_radii(centers):
    """Computes maximum valid radii for a fixed set of centers."""
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist, axis=1)
    r_pair = 0.5 * min_dist
    
    return np.minimum(r_bound, r_pair)

def obj_centers(c_flat):
    """Objective for basinhopping: minimize negative sum of implicit radii."""
    centers = c_flat.reshape(-1, 2)
    radii = compute_radii(centers)
    return -np.sum(radii)

def obj_full(vars_flat):
    """Objective for SLSQP refinement: minimize negative sum of explicit radii."""
    return -np.sum(vars_flat[2 * N_CIRCLES:])

def con_full(vars_flat):
    """Constraints for SLSQP: boundary and non-overlap."""
    n = N_CIRCLES
    centers = vars_flat[:2 * n].reshape(n, 2)
    radii = vars_flat[2 * n:]
    
    cons = [centers[:, 0] - radii, 1.0 - centers[:, 0] - radii, 
            centers[:, 1] - radii, 1.0 - centers[:, 1] - radii]
            
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append(d[mask] - (radii[:, np.newaxis] + radii[np.newaxis, :])[mask])
    
    return np.concatenate([c.flatten() for c in cons])

def accept_test(kwargs):
    """Acceptance criterion for basinhopping."""
    return kwargs["new_f"] < kwargs["f"]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    np.random.seed(42)
    
    starts = []
    
    # 1. Hexagonal lattice initialization
    c = []
    r_est = 0.105
    dy = r_est * math.sqrt(3)
    dx = 2 * r_est
    row = 0
    while len(c) < n:
        y = r_est + row * dy
        if y + r_est > 1.0: break
        offset = (dx / 2.0) if row % 2 == 1 else 0.0
        x = r_est + offset
        while x + r_est <= 1.0 and len(c) < n:
            c.append([x, y]); x += dx
        row += 1
    while len(c) < n: c.append(np.random.rand(2).tolist())
    starts.append(np.array(c[:n]).flatten())
    
    # 2. Dense grid initialization
    gx = np.linspace(0.1, 0.9, 6)
    gy = np.linspace(0.15, 0.85, 5)
    cx, cy = np.meshgrid(gx, gy)
    starts.append(np.column_stack((cx.flatten(), cy.flatten()))[:n].flatten())
    
    # 3. Random dense clusters
    for _ in range(4):
        c = np.random.rand(n, 2) * 0.8 + 0.1
        starts.append(c.flatten())
        
    best_sum = -np.inf
    best_centers = None
    
    minimizer_kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 6000, "xatol": 1e-8, "fatol": 1e-9}}
    
    # Global search phase
    for i, x0 in enumerate(starts):
        try:
            ret = basinhopping(obj_centers, x0, minimizer_kwargs=minimizer_kwargs, 
                               niter=60, stepsize=0.05, interval=50, 
                               accept_test=accept_test, seed=42+i)
            if -ret.fun > best_sum:
                best_sum = -ret.fun
                best_centers = ret.x.reshape(n, 2)
        except Exception:
            pass
            
    # Local constrained refinement phase
    if best_centers is not None:
        radii_init = compute_radii(best_centers)
        x0_full = np.concatenate([best_centers.flatten(), radii_init])
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        constraints = {'type': 'ineq', 'fun': con_full}
        try:
            res = minimize(obj_full, x0_full, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-12})
            curr_sum = -res.fun
            c_vals = con_full(res.x)
            if np.all(c_vals >= -1e-8) and curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x[:2 * n].reshape(n, 2)
                best_radii = res.x[2 * n:]
            else:
                best_radii = compute_radii(best_centers)
        except Exception:
            best_radii = compute_radii(best_centers)
    else:
        best_centers = starts[0].reshape(n, 2)
        best_radii = compute_radii(best_centers)
        
    # Final safety validation & adjustment
    centers = best_centers
    radii = best_radii
    for _ in range(10):
        valid = True
        for i in range(n):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if x - r < -1e-12 or x + r > 1.0 + 1e-12 or y - r < -1e-12 or y + r > 1.0 + 1e-12:
                valid = False; break
        if not valid: break
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False; break
            if not valid: break
        if valid: break
        radii *= 0.999
        
    return centers, radii, float(np.sum(radii))

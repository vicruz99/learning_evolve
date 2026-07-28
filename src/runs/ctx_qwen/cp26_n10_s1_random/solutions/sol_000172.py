# sol_000172 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000104 (state 0eb63b58) state=fdded374 sum of radii=2.259998 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp(centers):
    """
    Given fixed centers, solves the LP to maximize sum of radii.
    Constraints: r_i <= dist(i, boundary) and r_i + r_j <= dist(i, j)
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = [(0.0, None)] * n
    A_ub_list, b_ub_list = [], []
    
    # Boundary constraints
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(lim)
        
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(d)
            
    try:
        res = linprog(c_obj, A_ub=np.array(A_ub_list), b_ub=np.array(b_ub_list), 
                      bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def center_penalty(c_flat, radii, n, triu_i, triu_j):
    """
    Computes sum of squared constraint violations for fixed radii.
    Used to push centers apart to satisfy packing constraints.
    """
    c = c_flat.reshape(n, 2)
    p = 0.0
    
    # Boundary violations
    p += np.sum(np.maximum(0.0, radii - c[:, 0])**2)
    p += np.sum(np.maximum(0.0, radii - (1.0 - c[:, 0]))**2)
    p += np.sum(np.maximum(0.0, radii - c[:, 1])**2)
    p += np.sum(np.maximum(0.0, radii - (1.0 - c[:, 1]))**2)
    
    # Pairwise overlap violations
    dx = c[:, 0][:, None] - c[:, 0][None, :]
    dy = c[:, 1][:, None] - c[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = radii[:, None] + radii[None, :]
    
    p += np.sum(np.maximum(0.0, r_sum[triu_i, triu_j] - dist[triu_i, triu_j])**2)
    return p

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    triu_i, triu_j = np.triu_indices(n, k=1)
    bounds_c = [(0.0, 1.0)] * (2 * n)
    
    def make_hex(rows, r0=0.09):
        pts = []
        y = r0
        for ri, cnt in enumerate(rows):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
        while len(pts) < n:
            pts.append([0.5, 0.5])
        return np.array(pts[:n])

    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5]
    ]
    
    starts = []
    for p in patterns:
        if sum(p) < n: continue
        starts.append(make_hex(p))
        starts.append(make_hex(p, 0.095))
        
    for _ in range(8):
        starts.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    for cfg in starts:
        centers = cfg.copy()
        for step in range(25):
            radii, s = solve_lp(centers)
            if radii is None: break
            
            # Check strict validity and update best
            valid = True
            for i in range(n):
                x, y, r = centers[i, 0], centers[i, 1], radii[i]
                if x - r < -1e-9 or x + r > 1.0 + 1e-9 or y - r < -1e-9 or y + r > 1.0 + 1e-9:
                    valid = False; break
            if valid:
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                        if d < radii[i] + radii[j] - 1e-9:
                            valid = False; break
                    if not valid: break
            if valid and s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()
                
            # Optimize centers to reduce constraint violations
            res = minimize(center_penalty, centers.flatten(), args=(radii, n, triu_i, triu_j),
                           method='L-BFGS-B', bounds=bounds_c,
                           options={'maxiter': 3000, 'ftol': 1e-15})
            centers = res.x.reshape(n, 2)
            
            # Perturb to escape local minima in the penalty landscape
            centers += rng.uniform(-0.003, 0.003, centers.shape)
            centers = np.clip(centers, 0.0, 1.0)

    # Fallback if optimization fails
    if best_centers is None:
        best_centers = make_hex([5, 6, 5, 6, 4])
        best_radii, best_sum = solve_lp(best_centers)
        
    # Final safety scaling to guarantee strict validity for the checker's tolerance
    if best_centers is not None and best_radii is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            if r > 1e-9:
                scale = min(scale, x/r, (1.0 - x)/r, y/r, (1.0 - y)/r)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
                rs = best_radii[i] + best_radii[j]
                if rs > 1e-9:
                    scale = min(scale, d / rs)
        best_radii *= scale * 0.999999
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum

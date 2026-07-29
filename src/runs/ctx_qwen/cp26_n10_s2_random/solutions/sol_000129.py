# sol_000129 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000104 (state 8c305e91) state=15fc41c8 sum of radii=0.435756 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, 1)

def compute_radii(centers):
    """Computes the maximum valid radius for each circle given fixed centers."""
    x = centers[:, 0]
    y = centers[:, 1]
    # Maximum radius allowed by boundaries
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Maximum radius allowed by nearest neighbors
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(r_bound, r_pair)

def objective_centers(x):
    """Objective function for center optimization: maximize sum of exact radii."""
    c = x.reshape(N, 2).copy()
    # Keep centers strictly inside to prevent degenerate 0 radii
    c = np.clip(c, 1e-4, 1.0 - 1e-4)
    r = compute_radii(c)
    return -np.sum(r)

def generate_starts(num_starts):
    """Generates diverse initial configurations for multi-start optimization."""
    starts = []
    rng = np.random.default_rng(123)
    
    # Hexagonal lattice patterns known to be efficient for N~26
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 6, 5, 5, 4], [5, 5, 6, 6, 4], [4, 6, 5, 5, 6],
        [5, 6, 6, 5, 4], [6, 4, 6, 5, 5], [5, 5, 5, 6, 5],
        [6, 5, 5, 5, 5], [5, 5, 5, 5, 5, 1]
    ]
    
    for pat in patterns:
        if sum(pat) != N: 
            continue
        c = np.zeros((N, 2))
        idx = 0
        r_est = 0.095
        dy = np.sqrt(3) * r_est
        for r_idx, count in enumerate(pat):
            y = r_est + r_idx * dy
            offset = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + offset
            for _ in range(count):
                if idx < N:
                    c[idx] = [x, y]
                    idx += 1
                x += 2 * r_est
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.02, 0.98)
        starts.append(c.flatten())
        
    # Random dense starts
    for _ in range(max(10, num_starts // 2)):
        c = rng.uniform(0.1, 0.9, (N, 2))
        starts.append(c.flatten())
        
    # Corner-focused starts
    for _ in range(max(5, num_starts // 4)):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.06, 0.06], [0.94, 0.06], [0.06, 0.94], [0.94, 0.94]]
        for i in range(4):
            c[i] = corners[i]
        c += rng.normal(0, 0.012, c.shape)
        c = np.clip(c, 0.02, 0.98)
        starts.append(c.flatten())
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_sum = -np.inf
    
    center_bounds = [(0.001, 0.999)] * (2 * N)
    starts = generate_starts(45)
    
    # Phase 1: Powell optimization on centers
    for s_idx, x0 in enumerate(starts):
        try:
            # Initial run
            res = opt.minimize(objective_centers, x0, method='Powell',
                               bounds=center_bounds,
                               options={'maxiter': 2000, 'ftol': 1e-12})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(N, 2)
                
            # Extended run from best found so far
            if s_idx == 0 or (best_centers is not None and -res.fun > best_sum - 1e-5):
                res2 = opt.minimize(objective_centers, best_centers.flatten(), method='Powell',
                                    bounds=center_bounds,
                                    options={'maxiter': 4000, 'ftol': 1e-13})
                if -res2.fun > best_sum:
                    best_sum = -res2.fun
                    best_centers = res2.x.reshape(N, 2)
        except Exception:
            pass
            
    # Phase 2: Perturbation refinement to escape shallow local minima
    if best_centers is not None:
        for _ in range(25):
            pert = best_centers + rng.normal(0, 0.0025, best_centers.shape)
            pert = np.clip(pert, 0.01, 0.99)
            try:
                res = opt.minimize(objective_centers, pert.flatten(), method='Powell',
                                   bounds=center_bounds,
                                   options={'maxiter': 1500, 'ftol': 1e-12})
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_centers = res.x.reshape(N, 2)
            except Exception:
                pass
                
    # Compute exact maximal radii for optimized centers
    radii = compute_radii(best_centers)
    
    # Phase 3: High-precision SLSQP polish on joint variables
    def obj_joint(p):
        return -np.sum(p[2::3])
        
    def cons_joint(p):
        c = p[:2*N].reshape(N, 2)
        r = p[2*N:]
        # Boundary constraints
        con = []
        con.append(c[:, 0] - r)
        con.append(1.0 - c[:, 0] - r)
        con.append(c[:, 1] - r)
        con.append(1.0 - c[:, 1] - r)
        # Pairwise non-overlap
        d = np.linalg.norm(c[I_IDX] - c[J_IDX], axis=1)
        con.append(d - (r[I_IDX] + r[J_IDX]))
        return np.concatenate(con)
        
    bounds_joint = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    p0 = np.concatenate([best_centers.flatten(), radii])
    
    try:
        res = opt.minimize(obj_joint, p0, method='SLSQP', bounds=bounds_joint,
                           constraints={'type': 'ineq', 'fun': cons_joint},
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
        if -res.fun > best_sum - 1e-7:
            best_centers = res.x[:2*N].reshape(N, 2)
            radii = res.x[2*N:]
    except Exception:
        pass
        
    # Phase 4: Strict numerical repair to guarantee validator tolerance
    radii = radii.copy()
    for _ in range(40):
        changed = False
        # Clamp to boundaries
        for i in range(N):
            mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0],
                     best_centers[i, 1], 1.0 - best_centers[i, 1])
            if radii[i] > mx - 1e-10:
                radii[i] = mx
                changed = True
        # Resolve overlaps symmetrically
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(best_centers[i, 0] - best_centers[j, 0],
                               best_centers[i, 1] - best_centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    ov = radii[i] + radii[j] - d
                    radii[i] -= ov * 0.5
                    radii[j] -= ov * 0.5
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return best_centers, radii, final_sum

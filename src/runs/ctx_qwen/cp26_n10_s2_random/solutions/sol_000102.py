# sol_000102 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000069 (state 2c1a60b6) state=003649c1 sum of radii=1.076862 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26
i_idx, j_idx = np.triu_indices(N, k=1)
PAIR_COUNT = len(i_idx)

def compute_radii(centers):
    """Computes the maximum valid radius for each circle given fixed centers."""
    x, y = centers[:, 0], centers[:, 1]
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(r_bound, r_pair)

def obj_sum_radii(v):
    """Objective for center-only optimization: minimize negative sum of radii."""
    c = v.reshape(N, 2)
    r = compute_radii(c)
    return -np.sum(r)

def constraints(p):
    """
    Computes all boundary and non-overlap constraints for SLSQP.
    Returns a 1D array where each element must be >= 0.
    Uses squared distances for better gradient conditioning.
    """
    x, y, r = p[0::3], p[1::3], p[2::3]
    c = np.empty(4 * N + PAIR_COUNT)
    idx = 0
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c[idx:idx+N] = x - r; idx += N
    c[idx:idx+N] = 1.0 - x - r; idx += N
    c[idx:idx+N] = y - r; idx += N
    c[idx:idx+N] = 1.0 - y - r; idx += N
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dr = r[i_idx] + r[j_idx]
    c[idx:] = dx*dx + dy*dy - dr*dr
    return c

def repair(centers, radii):
    """Iteratively shrinks radii to strictly resolve overlaps and clamp to boundaries."""
    for _ in range(50):
        changed = False
        # Clamp to boundaries
        for i in range(N):
            max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                        centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > max_r - 1e-10:
                radii[i] = max_r
                changed = True
        # Resolve overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    ov = radii[i] + radii[j] - d
                    radii[i] -= ov * 0.5
                    radii[j] -= ov * 0.5
                    changed = True
        if not changed:
            break
    return radii

def get_bounds():
    """Creates variable bounds: x,y in [0,1], r in [1e-6, 0.5]"""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    return b

def generate_starts(num_starts):
    """Generates diverse initial configurations for multi-start optimization."""
    starts = []
    rng = np.random.default_rng(42)
    
    for s in range(num_starts):
        if s % 5 == 0:
            # Hexagonal lattice
            c = np.zeros((N, 2))
            idx = 0
            y = 0.09
            row = 0
            r = 0.09
            while idx < N:
                x_start = r if row % 2 == 0 else 2 * r
                x = x_start
                while x + r <= 1.0 + 1e-9 and idx < N:
                    c[idx] = [x, y]
                    idx += 1
                    x += 2 * r
                y += np.sqrt(3) * r
                row += 1
            c += rng.normal(0, 0.005, c.shape)
            c = np.clip(c, 0.05, 0.95)
        elif s % 5 == 1:
            # Perturbed 5x5 grid + 1 center
            gx = np.linspace(0.12, 0.88, 5)
            gy = np.linspace(0.12, 0.88, 5)
            cx, cy = np.meshgrid(gx, gy)
            c = np.column_stack((cx.flatten(), cy.flatten()))
            c = np.vstack([c, [0.5, 0.5]])
            c += rng.normal(0, 0.015, c.shape)
            c = np.clip(c, 0.05, 0.95)
        elif s % 5 == 2:
            # Dense random
            c = rng.uniform(0.1, 0.9, (N, 2))
        elif s % 5 == 3:
            # Corner-focused
            c = rng.uniform(0.15, 0.85, (N, 2))
            corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
            for i in range(4):
                c[i] = corners[i]
            c += rng.normal(0, 0.008, c.shape)
            c = np.clip(c, 0.02, 0.98)
        else:
            # Clustered random to encourage varying radii
            c = np.zeros((N, 2))
            for i in range(N):
                c[i] = rng.uniform(0.2, 0.8, 2)
            c += rng.normal(0, 0.01, c.shape)
            c = np.clip(c, 0.05, 0.95)
            
        starts.append(c.flatten())
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_centers = None
    best_sum = -np.inf
    
    # Phase 1: Multi-start SLSQP to find strong local optima
    starts = generate_starts(30)
    
    for s in starts:
        c_init = s.reshape(N, 2)
        r_init = compute_radii(c_init) * 0.95 # Start slightly feasible
        
        p0 = np.zeros(N * 3)
        p0[0::3] = c_init[:, 0]
        p0[1::3] = c_init[:, 1]
        p0[2::3] = r_init
        
        try:
            res = opt.minimize(lambda p: -np.sum(p[2::3]), p0, method='SLSQP', 
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-12})
            
            # Check feasibility and improve
            c_vals = constraints(res.x)
            if np.all(c_vals >= -1e-7):
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = res.x[:2*N].reshape(N, 2).copy()
        except Exception:
            continue
            
    # Fallback if optimization failed
    if best_centers is None:
        best_centers = starts[0].reshape(N, 2)
        
    # Phase 2: Powell refinement on centers only
    # This exploits the analytical radius function and handles non-smoothness well
    x0 = best_centers.flatten()
    center_bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        res_p = opt.minimize(obj_sum_radii, x0, method='Powell',
                             bounds=center_bounds,
                             options={'maxiter': 5000, 'ftol': 1e-13, 'xtol': 1e-13})
        if -res_p.fun > best_sum:
            best_centers = res_p.x.reshape(N, 2)
            best_sum = -res_p.fun
    except Exception:
        pass
        
    # Phase 3: Perturb & Polish to escape any remaining local minima
    rng = np.random.default_rng(123)
    for _ in range(10):
        pert = best_centers + rng.normal(0, 0.001, best_centers.shape)
        pert = np.clip(pert, 0.02, 0.98)
        try:
            res_p2 = opt.minimize(obj_sum_radii, pert.flatten(), method='Powell',
                                  bounds=center_bounds,
                                  options={'maxiter': 2000, 'ftol': 1e-12})
            if -res_p2.fun > best_sum:
                best_centers = res_p2.x.reshape(N, 2)
                best_sum = -res_p2.fun
        except Exception:
            continue
            
    # Compute final exact radii
    radii = compute_radii(best_centers)
    
    # Strict safety repair
    radii = repair(best_centers, radii)
    radii = np.maximum(radii, 0.0)
    
    # Final validation check against checker tolerances
    for _ in range(20):
        valid = True
        for i in range(N):
            x, y, r = best_centers[i, 0], best_centers[i, 1], radii[i]
            if x - r < -1e-12 or x + r > 1.0 + 1e-12 or y - r < -1e-12 or y + r > 1.0 + 1e-12:
                valid = False
                break
        if not valid:
            radii *= 0.9995
            continue
            
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                             best_centers[i, 1] - best_centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        radii *= 0.9995
        
    final_sum = float(np.sum(radii))
    return best_centers, radii, final_sum

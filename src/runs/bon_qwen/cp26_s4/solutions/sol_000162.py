# sol_000162 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2296af5d) state=c5b3286f sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (centers, radii, sum_radii) for 26 circles in a unit square.
    """
    n = 26
    
    def resolve_collisions(centers, radii, n_iter=100):
        """Simple force-directed repulsion to resolve overlaps."""
        for _ in range(n_iter):
            for i in range(n):
                # Boundary repulsion
                if centers[i, 0] - radii[i] < 0:
                    centers[i, 0] = radii[i]
                if centers[i, 0] + radii[i] > 1:
                    centers[i, 0] = 1 - radii[i]
                if centers[i, 1] - radii[i] < 0:
                    centers[i, 1] = radii[i]
                if centers[i, 1] + radii[i] > 1:
                    centers[i, 1] = 1 - radii[i]

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = centers[j] - centers[i]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Move apart proportional to overlap
                    move = (overlap / dist) * 0.5
                    centers[i] -= move * dist_vec
                    centers[j] += move * dist_vec

        # Final clamp
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])
        return centers

    def refine_pack(centers, radii):
        """Numerical optimization to maximize sum of radii."""
        def objective(x):
            # x contains [x1, y1, r1, x2, y2, r2, ...]
            # We want to maximize sum(r), so we minimize -sum(r)
            current_r = x[2::3]
            return -np.sum(current_r)

        def constraints_func(x):
            cons = []
            # Pairwise non-overlap
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    # dist^2 >= (ri + rj)^2  => dist^2 - (ri+rj)^2 >= 0
                    val = (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                    cons.append(val)
            # Boundary constraints: ri <= x <= 1-ri  => x - ri >= 0, 1 - x - ri >= 0
            for i in range(n):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                cons.append(xi - ri)
                cons.append(1 - xi - ri)
                cons.append(yi - ri)
                cons.append(1 - yi - ri)
                cons.append(ri) # r >= 0
            return cons

        # Flatten parameters
        x0 = np.hstack([centers, radii])
        
        # Define constraints for SLSQP
        n_cons = n * 4 + (n * (n - 1) // 2) + n
        cons = [{'type': 'ineq', 'fun': lambda x, idx=idx: constraints_func(x)[idx]} 
                for idx in range(n_cons)]

        # Use SLSQP to maximize sum of radii
        try:
            res = scipy.optimize.minimize(objective, x0, method='SLSQP', constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
            if res.success:
                x_opt = res.x
                centers = np.array([[x_opt[3*i], x_opt[3*i+1]] for i in range(n)])
                radii = np.array([x_opt[3*i+2] for i in range(n)])
                # Resolve numerical issues
                centers = resolve_collisions(centers, radii, 50)
                return centers, radii
        except Exception:
            pass
        return centers, radii

    best_centers = None
    best_radii = None
    best_sum = -1.0

    # Configuration 1: Hexagonal-like rows
    configs = []
    
    # 1. Random
    np.random.seed(42)
    c = np.random.rand(n, 2)
    r = np.full(n, 0.02)
    configs.append((c, r))

    # 2. 5x5 Grid + 1 small circle
    c = np.zeros((n, 2))
    r = np.full(n, 0.1)
    idx = 0
    for row in range(5):
        for col in range(5):
            c[idx, 0] = 0.1 + col * 0.2
            c[idx, 1] = 0.1 + row * 0.2
            idx += 1
    c[idx] = [0.3, 0.3]
    r[idx] = 0.04
    configs.append((c, r))

    # 3. Staggered Rows (5, 6, 5, 6, 4)
    c = np.zeros((n, 2))
    r = np.full(n, 0.08) # Start small to fit
    row_centers_y = [0.1, 0.1 + 0.08*np.sqrt(3), 0.1 + 2*0.08*np.sqrt(3), 0.1 + 3*0.08*np.sqrt(3), 0.1 + 4*0.08*np.sqrt(3)]
    row_counts = [5, 6, 5, 6, 4]
    idx = 0
    for row_idx, count in enumerate(row_counts):
        y = row_centers_y[row_idx]
        shift = 0.04 if row_idx % 2 == 1 else 0
        for col in range(count):
            x = 0.08 + shift + col * 0.16
            c[idx, 0] = x
            c[idx, 1] = y
            idx += 1
    configs.append((c, r))

    # 4. Hexagonal packing of 26 circles (optimized for space)
    c = np.zeros((n, 2))
    r = np.full(n, 0.06)
    k = 0
    # Rows with varying counts to fit in square
    # 5, 5, 5, 5, 5, 1 is 26
    row_counts = [5, 5, 5, 5, 5, 1]
    dy = 0.06 * np.sqrt(3)
    for i, count in enumerate(row_counts):
        y = 0.06 + i * dy
        shift = 0.06 if i % 2 == 1 else 0
        for j in range(count):
            x = 0.06 + shift + j * 0.12
            if k < n:
                c[k, 0] = x
                c[k, 1] = y
                k += 1
    configs.append((c, r))

    for init_c, init_r in configs:
        try:
            # Initial expansion
            centers = init_c.copy()
            radii = init_r.copy()
            step = 0.001
            for _ in range(200):
                radii += step
                centers = resolve_collisions(centers, radii, 20)
                # Reduce step size if stuck
                step *= 0.995
            
            # Refine
            centers, radii = refine_pack(centers, radii)
            
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        except Exception as e:
            pass

    # Final check and cleanup
    if best_centers is None:
        # Fallback to a safe grid
        best_centers = np.random.rand(n, 2) * 0.5 + 0.25
        best_radii = np.full(n, 0.01)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum

# Note: The function run_packing is defined as required.
# We provide the function to be called.

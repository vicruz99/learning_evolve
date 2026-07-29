# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state da2150ba) state=1db4001b sum of radii=2.617528 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize


def neg_sum_radii(v, n):
    """Objective: negative sum of radii (to minimize)"""
    return -np.sum(v[2 * n :])


def compute_constraints(v, n):
    """
    Compute all constraint values.
    All constraints should be >= 0.
    Uses squared distance to avoid square roots.
    """
    constraints = []
    for i in range(n):
        # x >= r
        constraints.append(v[2 * i] - v[2 * n + i])
        # x <= 1 - r
        constraints.append(1.0 - v[2 * i] - v[2 * n + i])
        # y >= r
        constraints.append(v[2 * i + 1] - v[2 * n + i])
        # y <= 1 - r
        constraints.append(1.0 - v[2 * i + 1] - v[2 * n + i])
        # r >= 0
        constraints.append(v[2 * n + i])
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[2 * i] - v[2 * j]
            dy = v[2 * i + 1] - v[2 * j + 1]
            dist_sq = dx * dx + dy * dy
            r_sum = v[2 * n + i] + v[2 * n + j]
            constraints.append(dist_sq - r_sum * r_sum)
    
    return np.array(constraints)


def init_hexagonal(n, row_sizes, scale=0.9):
    """Initialize circles in hexagonal pattern with given row sizes."""
    n_rows = len(row_sizes)
    
    # Compute feasible radius for this pattern
    max_cols = max(row_sizes)
    r_width = 1.0 / (2.0 * max_cols)
    r_height = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3))
    r = min(r_width, r_height) * scale
    
    centers = np.zeros((n, 2))
    radii = np.full(n, r)
    idx = 0
    
    for row in range(n_rows):
        for col in range(row_sizes[row]):
            x = r + col * 2.0 * r
            if row % 2 == 1:
                x += r
            y = r + row * r * np.sqrt(3)
            centers[idx] = [x, y]
            idx += 1
    
    return centers, radii


def init_grid(n, scale=0.85):
    """Initialize circles in a grid pattern."""
    # 5x5 grid + 1 extra
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
            idx += 1
    # Extra circle in a gap
    centers[25] = [0.2, 0.2]
    
    r = 0.09 * scale
    radii = np.full(n, r)
    radii[25] = 0.04 * scale
    
    return centers, radii


def init_perturbed_hexagonal(n, seed):
    """Initialize with perturbed hexagonal packing."""
    rng = np.random.RandomState(seed)
    
    row_configs = [
        [5, 5, 4, 5, 4, 3],
        [4, 5, 4, 5, 4, 4],
        [5, 4, 5, 4, 5, 3],
        [4, 4, 5, 4, 5, 4],
        [5, 5, 5, 4, 4, 3],
        [3, 5, 4, 5, 4, 5],
        [4, 5, 5, 4, 5, 3],
        [5, 4, 4, 5, 5, 3],
        [4, 4, 4, 5, 5, 4],
        [5, 5, 4, 4, 5, 3],
    ]
    
    config_idx = seed % len(row_configs)
    row_sizes = row_configs[config_idx]
    
    centers, radii = init_hexagonal(n, row_sizes, scale=0.88)
    
    # Add small random perturbation
    noise = rng.randn(n, 2) * 0.005
    centers += noise
    
    # Add small radius perturbation
    r_noise = rng.randn(n) * 0.002
    radii = np.maximum(radii + r_noise, 0.01)
    
    return centers, radii


def optimize_from_init(centers, radii, n):
    """Run optimization from given initial state."""
    x0 = np.concatenate([centers.flatten(), radii])
    
    cons = {
        'type': 'ineq',
        'fun': compute_constraints,
        'args': (n,)
    }
    
    result = minimize(
        neg_sum_radii,
        x0,
        args=(n,),
        method='SLSQP',
        constraints=cons,
        options={
            'maxiter': 3000,
            'ftol': 1e-14,
            'disp': False
        }
    )
    
    if result.success or result.fun < 0:
        centers_opt = result.x[:2 * n].reshape(n, 2)
        radii_opt = result.x[2 * n :]
        radii_opt = np.maximum(radii_opt, 1e-12)
        return centers_opt, radii_opt, -result.fun, True
    
    # Even if not fully successful, return best found
    centers_opt = result.x[:2 * n].reshape(n, 2)
    radii_opt = np.maximum(result.x[2 * n :], 1e-12)
    return centers_opt, radii_opt, np.sum(radii_opt), result.success


def validate_packing(centers, radii):
    """Validate packing - copied from problem statement."""
    n = centers.shape[0]
    
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    
    return True


def refine_packing(centers, radii, n, iterations=100):
    """
    Greedy refinement: try to increase each circle's radius individually,
    then adjust positions to resolve conflicts.
    """
    centers = centers.copy()
    radii = radii.copy()
    
    for _ in range(iterations):
        improved = False
        for i in range(n):
            # Try increasing radius
            old_r = radii[i]
            for delta in np.linspace(0.001, 0.02, 5):
                radii[i] = old_r + delta
                valid = True
                
                # Check boundary
                if (centers[i, 0] - radii[i] < -1e-10 or 
                    centers[i, 0] + radii[i] > 1 + 1e-10 or
                    centers[i, 1] - radii[i] < -1e-10 or
                    centers[i, 1] + radii[i] > 1 + 1e-10):
                    valid = False
                    radii[i] = old_r
                    break
                
                # Check overlaps
                for j in range(n):
                    if i == j:
                        continue
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[i] + radii[j] - 1e-10:
                        valid = False
                        radii[i] = old_r
                        break
                
                if valid:
                    improved = True
                    break
                else:
                    radii[i] = old_r
        
        if not improved:
            break
    
    return centers, radii


def run_packing():
    """Main function to pack 26 circles in unit square."""
    n = 26
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Try multiple initializations
    seeds = list(range(30))
    
    for seed in seeds:
        centers_init, radii_init = init_perturbed_hexagonal(n, seed)
        centers_opt, radii_opt, s, success = optimize_from_init(centers_init, radii_init, n)
        
        # Refine
        centers_refined, radii_refined = refine_packing(centers_opt, radii_opt, n, iterations=50)
        s_refined = np.sum(radii_refined)
        
        if s_refined > s:
            centers_opt = centers_refined
            radii_opt = radii_refined
            s = s_refined
        
        if s > best_sum:
            best_sum = s
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
    
    # Additional refinement pass on the best solution
    best_centers, best_radii = refine_packing(best_centers, best_radii, n, iterations=200)
    best_sum = np.sum(best_radii)
    
    # Final optimization pass on the refined solution
    if best_centers is not None:
        centers_opt2, radii_opt2, s2, _ = optimize_from_init(best_centers, best_radii, n)
        if s2 > best_sum:
            best_sum = s2
            best_centers = centers_opt2
            best_radii = radii_opt2
        best_centers, best_radii = refine_packing(best_centers, best_radii, n, iterations=100)
        best_sum = np.sum(best_radii)
    
    # Ensure radii are non-negative
    if best_radii is not None:
        best_radii = np.maximum(best_radii, 0)
    
    # Final validation check
    if validate_packing(best_centers, best_radii):
        return best_centers, best_radii, best_sum
    
    # Fallback: use a simple grid if validation fails
    centers_fallback = np.zeros((n, 2))
    radii_fallback = np.full(n, 0.09)
    idx = 0
    for i in range(5):
        for j in range(5):
            centers_fallback[idx] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
            idx += 1
    centers_fallback[25] = [0.2, 0.2]
    radii_fallback[25] = 0.04
    
    return centers_fallback, radii_fallback, np.sum(radii_fallback)

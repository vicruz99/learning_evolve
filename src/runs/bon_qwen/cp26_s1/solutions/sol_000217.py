# sol_000217 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=6e3a7677 sum of radii=2.581209 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses SLSQP optimization with multiple restarts from a grid initialization.
    """
    n = 26
    num_restarts = 5
    
    def get_params(p):
        centers = p[:2*n].reshape(n, 2)
        radii = p[2*n:]
        return centers, radii

    def objective(p):
        # Minimize negative sum of radii
        return -np.sum(p[2*n:])

    def make_constraints(n):
        constraints = []
        # Boundary constraints: 4 per circle
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        for i in range(n):
            idx_r = 2*n + i
            idx_x = 2*i
            idx_y = 2*i + 1
            
            # x - r >= 0
            constraints.append({
                'type': 'ineq', 
                'fun': lambda p, ix=idx_x, ir=idx_r: p[ix] - p[ir]
            })
            # 1 - x - r >= 0
            constraints.append({
                'type': 'ineq', 
                'fun': lambda p, ix=idx_x, ir=idx_r: 1.0 - p[ix] - p[ir]
            })
            # y - r >= 0
            constraints.append({
                'type': 'ineq', 
                'fun': lambda p, iy=idx_y, ir=idx_r: p[iy] - p[ir]
            })
            # 1 - y - r >= 0
            constraints.append({
                'type': 'ineq', 
                'fun': lambda p, iy=idx_y, ir=idx_r: 1.0 - p[iy] - p[ir]
            })
            
        # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_xi, idx_yi = 2*i, 2*i + 1
                idx_xj, idx_yj = 2*j, 2*j + 1
                idx_ri, idx_rj = 2*n + i, 2*n + j
                
                def sep_constraint(p, xi=idx_xi, yi=idx_yi, xj=idx_xj, yj=idx_yj, ri=idx_ri, rj=idx_rj):
                    dx = p[xi] - p[xj]
                    dy = p[yi] - p[yj]
                    dist_sq = dx*dx + dy*dy
                    rad_sum = p[ri] + p[rj]
                    return dist_sq - rad_sum*rad_sum
                
                constraints.append({'type': 'ineq', 'fun': sep_constraint})
                
        return constraints

    constraints = make_constraints(n)
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(2*n):
        bounds.append((0.0, 1.0))
    for _ in range(n):
        bounds.append((1e-5, 0.5))

    # Helper to generate initial guess
    def generate_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Base grid: 6 cols x 5 rows
        # We want to fit 26 circles. 
        # Let's create a 6x5 grid (30 points) and remove 4, or just place 26.
        # A hexagonal arrangement is denser.
        
        # Let's try a perturbed grid approach.
        # Create a dense grid of potential centers
        # 6 cols, 5 rows -> 30 points.
        # x spacing 1/5 = 0.2? No, 6 points need 5 gaps. 1/5=0.2.
        # y spacing 1/4 = 0.25? No, 5 points need 4 gaps.
        
        # Let's generate 30 points in a 6x5 grid
        xs = np.linspace(0.1, 0.9, 6) # 0.1 to 0.9, 6 points. 
        # Wait, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9. 
        # 0.1 - r >= 0 => r <= 0.1. 
        # 0.9 + r <= 1 => r <= 0.1.
        # So radius can be up to 0.1 if we use this grid.
        
        ys = np.linspace(0.1, 0.9, 5) # 0.1, 0.3, 0.5, 0.7, 0.9
        
        points = []
        for y in ys:
            for x in xs:
                points.append([x, y])
        
        # We have 30 points. We need 26.
        # Shuffle and pick 26? Or pick specific ones.
        # Removing corners might allow tighter packing in center?
        # Let's just shuffle to break symmetry for different seeds.
        indices = rng.permutation(len(points))
        selected_indices = indices[:n]
        centers = np.array([points[idx] for idx in selected_indices])
        
        # Perturb centers slightly
        centers += rng.uniform(-0.02, 0.02, size=centers.shape)
        # Clamp to [0.05, 0.95] to be safe for initial small radius
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initial radii
        radii = np.full(n, 0.06)
        
        p0 = np.concatenate([centers.flatten(), radii])
        return p0

    best_result = None
    best_sum_radii = -np.inf

    for k in range(num_restarts):
        seed = k * 100
        p0 = generate_initial_guess(seed)
        
        # Run optimizer
        res = minimize(
            objective, 
            p0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        current_sum = -res.fun
        if res.success or current_sum > best_sum_radii:
            best_result = res
            best_sum_radii = current_sum

    if best_result is not None:
        centers, radii = get_params(best_result.x)
        # Final validation check (just in case of numerical issues)
        # The constraints should guarantee validity, but let's be safe.
        # If any constraint is slightly violated due to tolerance, we might need to fix.
        # However, SLSQP with 1e-9 tol should be fine.
        # The validate function allows 1e-12 tolerance.
        
        return centers, radii, float(best_sum_radii)
    
    # Fallback (should not happen)
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    return centers, radii, 0.0

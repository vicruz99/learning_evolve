# sol_000341 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bd19375) state=97197a46 sum of radii=2.605630 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal lattice initialization and SLSQP optimization.
    """
    n_circles = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Helper function to create initial hexagonal grid configuration
    def get_initial_config(seed):
        rng = np.random.RandomState(seed)
        
        # Estimate a starting radius. 
        # Square grid 5x5 -> r=0.1. Hexagonal should allow slightly more.
        # Start slightly smaller to ensure feasibility.
        r_start = 0.09 
        
        # Generate hexagonal points
        # Spacing horizontally: 2*r, vertically: sqrt(3)*r
        # We generate a dense set of potential centers
        points = []
        # Cover a bit more than [0,1] to allow shifting
        x_range = np.arange(0, 1.5, 2 * r_start)
        y_range = np.arange(0, 1.5, np.sqrt(3) * r_start)
        
        for i, y in enumerate(y_range):
            offset = r_start if i % 2 == 1 else 0
            for x in x_range:
                cx = x + offset
                if 0 <= cx <= 1 and 0 <= y <= 1:
                    points.append([cx, y])
        
        points = np.array(points)
        
        # If we have fewer points than needed, reduce radius and retry (simple fallback)
        # But with r=0.09 we should have plenty.
        # Select n_circles points. 
        # Prefer points closer to center or just random subset?
        # Random subset helps diversity.
        if len(points) >= n_circles:
            idx = rng.choice(len(points), n_circles, replace=False)
            centers = points[idx].copy()
        else:
            # Fallback to random if grid is sparse (unlikely with r=0.09)
            centers = rng.rand(n_circles, 2)
        
        # Add small random perturbation to break symmetry
        centers += rng.uniform(-0.01, 0.01, centers.shape)
        centers = np.clip(centers, 0.01, 0.99)
        
        # Initial radii
        radii = np.full(n_circles, r_start)
        
        return centers, radii

    # Optimization wrapper
    def optimize_config(centers, radii):
        n = len(radii)
        
        # Variables: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([
                (0.0, 1.0), # x
                (0.0, 1.0), # y
                (0.0, 0.5)  # r
            ])

        # Objective: Maximize sum(r) => Minimize -sum(r)
        def objective(vars_arr):
            r = vars_arr[2::3]
            return -np.sum(r)

        # Constraints
        # 1. Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        #    => x - r >= 0, 1 - x - r >= 0, etc.
        def boundary_constraints(vars_arr):
            cons = []
            for i in range(n):
                x = vars_arr[3*i]
                y = vars_arr[3*i+1]
                r = vars_arr[3*i+2]
                cons.append(x - r)
                cons.append(1.0 - x - r)
                cons.append(y - r)
                cons.append(1.0 - y - r)
            return np.array(cons)

        # 2. Non-overlap: dist(i,j) >= r_i + r_j
        #    dist^2 >= (r_i + r_j)^2  <=> dist - (r_i + r_j) >= 0
        #    Using dist - sum_r >= 0 is better for smoothness than squared?
        #    Actually dist is sqrt, non-differentiable at 0, but we stay away from 0.
        #    Using squared form: dist_sq - (r_i+r_j)^2 >= 0 is valid but involves squares.
        #    Let's use linearized distance constraint for stability: dist - (r_i+r_j) >= 0
        def non_overlap_constraints(vars_arr):
            cons = []
            centers = vars_arr.reshape(-1, 3)[:, :2]
            radii = vars_arr.reshape(-1, 3)[:, 2]
            
            # Vectorized computation for speed
            # centers: (n, 2), radii: (n,)
            # diff: (n, n, 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2) # (n, n)
            
            # sum_radii matrix
            rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # We only need upper triangle
            for i in range(n):
                for j in range(i + 1, n):
                    cons.append(dists[i, j] - rad_sum[i, j])
            return np.array(cons)

        cons = []
        cons.append({'type': 'ineq', 'fun': boundary_constraints})
        cons.append({'type': 'ineq', 'fun': non_overlap_constraints})

        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'ftol': 1e-12, 'maxiter': 200, 'disp': False})
            if res.success:
                final_centers = res.x.reshape(-1, 3)[:, :2]
                final_radii = res.x.reshape(-1, 3)[:, 2]
                return final_centers, final_radii, -res.fun
            else:
                return centers, radii, np.sum(radii) # Return original if failed
        except Exception:
            return centers, radii, np.sum(radii)

    # Try multiple seeds to find global optimum
    seeds = range(10) 
    for seed in seeds:
        centers_init, radii_init = get_initial_config(seed)
        c, r, s = optimize_config(centers_init, radii_init)
        if s > best_sum_radii:
            best_sum_radii = s
            best_centers = c
            best_radii = r
            
    # Final validation and slight cleanup
    # Ensure strictly positive radii and inside bounds
    if best_radii is not None:
        best_radii = np.maximum(best_radii, 1e-6)
        # Clamp centers to ensure r constraint holds numerically
        for i in range(26):
            r = best_radii[i]
            best_centers[i, 0] = np.clip(best_centers[i, 0], r, 1-r)
            best_centers[i, 1] = np.clip(best_centers[i, 1], r, 1-r)
            # Recalculate radius to be safe? No, just ensure validity.
            # If clipping changed center, r might be invalid w.r.t other circles?
            # But clipping moves center inside, increasing distance to boundaries,
            # but might decrease distance to other centers.
            # However, the optimizer should have found a feasible point.
            # The clipping is just for numerical safety.
            
            # Actually, if we clip center, we might violate non-overlap.
            # But the optimizer output should be valid.
            # Let's just return the optimized result.
            
    return best_centers, best_radii, best_sum_radii

# To execute and check
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    # Validation call (not included in output but for logic check)
    # print(validate_packing(centers, radii))

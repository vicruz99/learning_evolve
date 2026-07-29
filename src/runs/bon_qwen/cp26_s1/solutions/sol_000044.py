# sol_000044 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f1389a1) state=416e88cc sum of radii=0.657180 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the 26-circle packing problem to maximize the sum of radii.
    """
    n_circles = 26
    
    # Helper to compute distance matrix between centers
    def get_dist_matrix(centers):
        # centers shape (n, 2)
        # returns (n, n) matrix of distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def evaluate_solution(centers, radii):
        """
        Returns a score. Higher is better.
        Penalizes overlaps and boundary violations.
        """
        n = centers.shape[0]
        sum_radii = np.sum(radii)
        penalty = 0.0
        
        # Boundary constraints: 0 <= x-r, y-r and x+r <= 1, y+r <= 1
        # Equivalent to: r <= x <= 1-r and r <= y <= 1-r
        # Penalty if violated
        x, y = centers[:, 0], centers[:, 1]
        r = radii
        
        # Lower bound violations
        penalty += 1000.0 * np.sum(np.maximum(0, r - x))
        penalty += 1000.0 * np.sum(np.maximum(0, r - y))
        # Upper bound violations
        penalty += 1000.0 * np.sum(np.maximum(0, x + r - 1.0))
        penalty += 1000.0 * np.sum(np.maximum(0, y + r - 1.0))
        
        # Overlap constraints
        # dist(i, j) >= r_i + r_j
        # We only need to check i < j
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Strong penalty for overlaps
                    penalty += 10000.0 * overlap 
        
        # If penalty is 0, solution is valid.
        # We want to maximize sum_radii, so we return -sum_radii + penalty
        # Actually, to make it clear, let's return a value to MINIMIZE.
        # Objective: -sum_radii + penalty
        return -sum_radii + penalty

    def objective(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        # Shape (78,)
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        for i in range(n_circles):
            idx = i * 3
            centers[i, 0] = vars[idx]
            centers[i, 1] = vars[idx+1]
            radii[i] = vars[idx+2]
        return evaluate_solution(centers, radii)

    best_score = float('inf')
    best_centers = None
    best_radii = None

    # Try multiple random starts
    for seed in range(10):
        np.random.seed(seed)
        
        # Initialization: Hexagonal-ish grid
        centers_init = np.zeros((n_circles, 2))
        radii_init = np.full(n_circles, 0.10) # Start with slightly optimistic radius
        
        # Place in a rough grid
        idx = 0
        row = 0
        col = 0
        step_x = 0.21
        step_y = 0.18
        
        while idx < n_circles:
            # Alternate row offset
            offset = 0.105 if row % 2 == 1 else 0.0
            x = 0.105 + col * step_x + offset
            y = 0.105 + row * step_y * np.sqrt(3)/2 # Hex vertical spacing approx
            
            if idx < n_circles:
                centers_init[idx, 0] = x
                centers_init[idx, 1] = y
                idx += 1
            else:
                break
            
            col += 1
            if col > 5: # Approx 5-6 per row
                col = 0
                row += 1
        
        # Flatten for optimizer
        vars_init = np.zeros(3 * n_circles)
        for i in range(n_circles):
            vars_init[i*3] = centers_init[i, 0]
            vars_init[i*3+1] = centers_init[i, 1]
            vars_init[i*3+2] = radii_init[i]

        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = [(0.0, 1.0) for _ in range(3 * n_circles)]
        # Tighten radius bounds slightly to avoid silly solutions
        for i in range(n_circles):
            bounds[i*3+2] = (0.0, 0.5)

        try:
            res = opt.minimize(objective, vars_init, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 2000, 'ftol': 1e-9})
            
            if res.fun < best_score:
                # Check if valid (penalty approx 0)
                # Reconstruct
                c = np.zeros((n_circles, 2))
                r = np.zeros(n_circles)
                for i in range(n_circles):
                    c[i, 0] = res.x[i*3]
                    c[i, 1] = res.x[i*3+1]
                    r[i] = res.x[i*3+2]
                
                # Quick validity check
                valid = True
                # Check bounds
                for i in range(n_circles):
                    if r[i] < 0 or c[i, 0] < -1e-6 or c[i, 0] > 1+1e-6 or \
                       c[i, 1] < -1e-6 or c[i, 1] > 1+1e-6 or \
                       c[i, 0] - r[i] < -1e-6 or c[i, 0] + r[i] > 1+1e-6 or \
                       c[i, 1] - r[i] < -1e-6 or c[i, 1] + r[i] > 1+1e-6:
                        valid = False
                        break
                # Check overlaps
                if valid:
                    dists = get_dist_matrix(c)
                    for i in range(n_circles):
                        for j in range(i+1, n_circles):
                            if dists[i, j] < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    best_score = res.fun
                    best_centers = c
                    best_radii = r

        except Exception as e:
            print(f"Optimization failed with seed {seed}: {e}")
            continue

    if best_centers is None:
        # Fallback to a simple grid if optimization fails
        best_centers = np.zeros((26, 2))
        best_radii = np.zeros(26)
        r = 0.1
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < 26:
                    best_centers[idx] = [0.1 + j*0.2, 0.1 + i*0.2]
                    best_radii[idx] = r
                    idx += 1

    sum_radii = np.sum(best_radii)
    return best_centers, best_radii, sum_radii

# To test locally
if __name__ == "__main__":
    import numpy as np
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any(): return False
        if np.isnan(radii).any(): return False
        for i in range(n):
            if radii[i] < 0: return False
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

    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")

# sol_000123 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 69804dab) state=86f5ab0f sum of radii=1.641421 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    def get_radius(centers, i):
        x, y = centers[i]
        # Distance to boundaries
        r_bound = min(x, 1.0 - x, y, 1.0 - y)
        # Distance to other circles
        dists = np.linalg.norm(centers - centers[i], axis=1)
        dists[i] = np.inf # Ignore self
        r_circles = np.min(dists) / 2.0
        return min(r_bound, r_circles)

    def objective(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        # Smooth approximation of min function using log-sum-exp to avoid NaN gradients
        alpha = 100.0
        r_total = 0.0
        for i in range(n_circles):
            x, y = centers[i]
            bound_vals = [x, 1.0 - x, y, 1.0 - y]
            
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            circ_vals = dists / 2.0
            
            all_vals = bound_vals + list(circ_vals)
            
            # Stable log-sum-exp for minimum
            max_val = np.max(all_vals)
            sum_exp = np.sum(np.exp(-alpha * (np.array(all_vals) - max_val)))
            r_i = max_val - (1.0 / alpha) * np.log(sum_exp)
            r_total += r_i
        return -r_total  # Minimize negative sum

    def get_gradient(centers):
        grad = np.zeros_like(centers)
        for i in range(n_circles):
            x, y = centers[i]
            
            # Calculate active bottleneck
            bound_vals = np.array([x, 1.0 - x, y, 1.0 - y])
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            circ_vals = dists / 2.0
            
            all_vals = np.concatenate([bound_vals, circ_vals])
            min_val = np.min(all_vals)
            
            # Identify active constraints (close to min)
            active_idx = np.isclose(all_vals, min_val, atol=1e-4)
            if not np.any(active_idx):
                active_idx[0] = True
            
            active_vals = all_vals[active_idx]
            r_i = np.mean(active_vals)
            
            # Determine gradient direction for active constraints
            # Boundary gradients
            dx, dy = 0.0, 0.0
            for k, val in enumerate(all_vals):
                if not active_idx[k]: continue
                
                if k == 0: dx += 1.0 # x = 0 -> r = x
                elif k == 1: dx -= 1.0 # x = 1 -> r = 1-x
                elif k == 2: dy += 1.0 # y = 0 -> r = y
                elif k == 3: dy -= 1.0 # y = 1 -> r = 1-y
                else:
                    j = k - 4
                    # Circle gradient
                    vec = centers[i] - centers[j]
                    dist = np.linalg.norm(vec)
                    if dist > 1e-9:
                        dx += vec[0] / (2.0 * dist)
                        dy += vec[1] / (2.0 * dist)
            
            # Normalize gradient magnitude if multiple active
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-9:
                grad[i] = np.array([dx, dy]) / norm
                
        return grad.flatten()

    def simulate(n_iter=5000):
        # Hexagonal grid initialization
        centers = np.zeros((n_circles, 2))
        idx = 0
        for row in range(8):
            for col in range(6):
                if idx >= n_circles: break
                x = (col + 0.5) * 0.15
                y = row * 0.15 * np.sqrt(3) + 0.05
                if x < 1.0 and y < 1.0:
                    centers[idx] = [x, y]
                    idx += 1

        # Random perturbation
        centers += np.random.uniform(-0.05, 0.05, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)

        # Optimization
        res = minimize(objective, centers.flatten(), method='L-BFGS-B', jac=get_gradient,
                       bounds=[(0.01, 0.99)] * (2 * n_circles), options={'maxiter': 1000})
        
        return res.x.reshape(-1, 2), -res.fun

    for _ in range(10):
        try:
            centers, total = simulate()
            if total > best_sum:
                best_sum = total
                best_centers = centers
        except Exception:
            continue

    if best_centers is None:
        # Fallback to grid
        best_centers = np.array([(i/5.0, j/5.0) for i in range(5) for j in range(5)]).reshape(25, 2)
        best_centers = np.vstack([best_centers, [[0.1, 0.1]]])
        best_sum = 0.0

    # Final radii calculation
    radii = np.array([get_radius(best_centers, i) for i in range(n_circles)])
    actual_sum = np.sum(radii)
    
    return best_centers, radii, actual_sum

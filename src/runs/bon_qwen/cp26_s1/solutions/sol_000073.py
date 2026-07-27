# sol_000073 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfef56bb) state=9d6f9f0f sum of radii=1.182398 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def solve_packing(n_circles, max_iter=200, step_size=0.005, n_restarts=10):
    """
    Optimizes circle packing to maximize sum of radii.
    """
    best_centers = None
    best_radii = None
    best_sum = -1.0

    # Indices for constraints to map duals back
    # We will construct the LP inside the loop
    
    for restart in range(n_restarts):
        # Initialize centers randomly
        # Place them with some margin to start valid
        centers = np.random.uniform(0.2, 0.8, size=(n_circles, 2))
        
        # Step size for gradient ascent
        lr = step_size
        
        for iteration in range(max_iter):
            # --- Step 1: Solve LP for Radii ---
            # Variables: r_0, ..., r_{n-1}
            # Maximize sum(r_i) => Minimize -sum(r_i)
            c_obj = -np.ones(n_circles)
            
            # Constraints: A_ub @ r <= b_ub
            # 1. r_i + r_j <= dist(i, j)
            # 2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
            # 3. r_i >= 0 (handled by bounds)
            
            m_pairs = n_circles * (n_circles - 1) // 2
            m_bounds = 4 * n_circles
            total_constraints = m_pairs + m_bounds
            
            A_ub = np.zeros((total_constraints, n_circles))
            b_ub = np.zeros(total_constraints)
            
            # Pairwise distance constraints
            idx = 0
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    A_ub[idx, i] = 1.0
                    A_ub[idx, j] = 1.0
                    b_ub[idx] = dist
                    idx += 1
            
            # Boundary constraints
            for i in range(n_circles):
                x, y = centers[i]
                # r <= x
                A_ub[idx, i] = 1.0
                b_ub[idx] = x
                idx += 1
                # r <= 1 - x
                A_ub[idx, i] = 1.0
                b_ub[idx] = 1.0 - x
                idx += 1
                # r <= y
                A_ub[idx, i] = 1.0
                b_ub[idx] = y
                idx += 1
                # r <= 1 - y
                A_ub[idx, i] = 1.0
                b_ub[idx] = 1.0 - y
                idx += 1
            
            # Bounds: r >= 0
            bounds = [(0, None) for _ in range(n_circles)]
            
            # Solve LP
            try:
                res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if not res.success:
                    # If LP fails, break or reset? 
                    # Usually shouldn't happen if centers are valid.
                    break
                
                current_radii = res.x
                current_sum = np.sum(current_radii)
                
                # Store best
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = current_radii.copy()
                
                # --- Step 2: Update Centers using Dual Variables ---
                # Dual variables (marginals) correspond to inequality constraints A_ub @ r <= b_ub
                # Since we minimized -sum(r), the duals relate to the objective increase.
                # For max problem, duals indicate sensitivity of optimal value to RHS (b_ub).
                # If b_ub increases (distance increases), objective (sum r) increases.
                # Dual value y_k >= 0 means increasing b_k by 1 increases obj by y_k.
                
                try:
                    duals = res.ineqlin.marginals
                except AttributeError:
                    # Fallback if attribute missing or version issue
                    duals = np.zeros(total_constraints)

                # Compute gradient for centers
                grad_centers = np.zeros_like(centers)
                
                # Contribution from pairwise constraints
                idx = 0
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dual_val = duals[idx]
                        if dual_val > 1e-9: # Only active constraints matter
                            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                            if dist > 1e-9:
                                # Gradient of dist w.r.t centers[i] is (centers[i] - centers[j]) / dist
                                # Increasing dist increases b_ub, which increases objective by dual_val
                                # So we want to move centers[i] away from centers[j]
                                dir_vec = (centers[i] - centers[j]) / dist
                                grad_centers[i] += dual_val * dir_vec
                                grad_centers[j] -= dual_val * dir_vec # Move j away from i
                        idx += 1
                
                # Contribution from boundary constraints
                # Constraints order: x, 1-x, y, 1-y for each i
                # r <= x => b = x. Gradient of b w.r.t x is 1. 
                # Increasing x helps. Force in +x direction.
                # r <= 1-x => b = 1-x. Gradient of b w.r.t x is -1.
                # Decreasing x helps. Force in -x direction.
                
                idx = m_pairs
                for i in range(n_circles):
                    # r <= x_i
                    grad_centers[i, 0] += duals[idx] 
                    idx += 1
                    # r <= 1 - x_i
                    grad_centers[i, 0] -= duals[idx]
                    idx += 1
                    # r <= y_i
                    grad_centers[i, 1] += duals[idx]
                    idx += 1
                    # r <= 1 - y_i
                    grad_centers[i, 1] -= duals[idx]
                    idx += 1
                
                # Normalize gradient to prevent huge steps
                grad_norm = np.linalg.norm(grad_centers)
                if grad_norm > 1e-9:
                    grad_centers = grad_centers / grad_norm
                
                # Update centers
                centers += lr * grad_centers
                
                # Project centers to [0, 1]
                # Actually, strict projection to [0,1] is needed.
                # But we also need to ensure centers are not "too close" to boundaries if radii are large?
                # No, the LP handles radii limits. Centers just need to be in [0,1].
                # However, if center moves outside [0,1], LP might become infeasible or r=0.
                # Let's clamp centers to [0, 1].
                centers = np.clip(centers, 0.0, 1.0)
                
                # Decay learning rate slightly?
                # lr *= 0.999 
                
            except Exception as e:
                # If LP fails or other error, stop this restart
                break

    return best_centers, best_radii, best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Main function to run the packing optimization.
    """
    n = 26
    # Use a larger number of iterations and restarts for better solution
    centers, radii, sum_radii = solve_packing(n, max_iter=300, step_size=0.01, n_restarts=15)
    
    # Final validation check (optional, but good for safety)
    # If radii are valid, return. 
    # Note: The LP ensures r_i + r_j <= dist, so no overlap.
    # The LP ensures r_i <= dist_to_boundary, so inside square.
    
    return centers, radii, float(sum_radii)

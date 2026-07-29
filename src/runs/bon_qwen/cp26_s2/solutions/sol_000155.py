# sol_000155 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 46a34d55) state=6d9f7121 sum of radii=1.710060 correctness=1.0
# stdout(first 200): Attempt 1: Sum = 1.62960 Attempt 2: Sum = 1.63205 Attempt 3: Sum = 1.63856 Attempt 4: Sum = 1.65600 Attempt 5: Sum = 1.71006
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    np.random.seed(42) # For reproducibility

    def solve_radii(centers):
        """
        Given fixed centers, solve LP to maximize sum of radii.
        Maximize sum(r_i)
        s.t.
          r_i + r_j <= dist(i, j)
          r_i <= x_i
          r_i <= 1 - x_i
          r_i <= y_i
          r_i <= 1 - y_i
          r_i >= 0
        """
        # Coefficients for objective: maximize sum(r_i) => minimize -sum(r_i)
        c = -np.ones(n)

        # Inequality constraints A_ub @ r <= b_ub
        # Constraints of form r_i + r_j <= d_ij
        # And boundary constraints r_i <= bound
        
        A_ub = []
        b_ub = []

        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                dist = np.linalg.norm(centers[i] - centers[j])
                b_ub.append(dist)
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            # r_i <= x
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(x)
            
            # r_i <= 1 - x
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0 - x)
            
            # r_i <= y
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(y)
            
            # r_i <= 1 - y
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0 - y)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Bounds for r_i: r_i >= 0
        bounds = [(0, None) for _ in range(n)]

        try:
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return -res.fun, res.x
            else:
                # Fallback if LP fails (should not happen with feasible centers)
                return 0.0, np.zeros(n)
        except Exception:
            return 0.0, np.zeros(n)

    def generate_initial_centers(method='hex'):
        """Generate initial centers based on a pattern."""
        centers = np.zeros((n, 2))
        
        if method == 'hex':
            # Try to fit in a hexagonal pattern
            # 5 rows. Counts: 6, 5, 6, 5, 4 sums to 26? 6+5+6+5+4 = 26.
            # Let's try to distribute.
            row_counts = [6, 5, 6, 5, 4]
            current_idx = 0
            row_height = 1.0 / 6.0 # Approx spacing
            
            for r_idx, count in enumerate(row_counts):
                y = 0.5 + (r_idx - 2) * (1.0 / 4.0) # Spread rows vertically
                # Shift odd rows
                offset = 0.05 if r_idx % 2 != 0 else 0.0
                
                # Calculate x spacing
                # width needed for count circles: 2*r + (count-1)*2r ? 
                # Just spread them evenly in [0, 1]
                xs = np.linspace(0.05 + offset, 0.95 - offset, count)
                
                for c_idx in range(count):
                    if current_idx < n:
                        centers[current_idx] = [xs[c_idx], y]
                        current_idx += 1
            return centers
        
        elif method == 'grid':
            # 5x5 grid plus one
            idx = 0
            for r in range(5):
                for c in range(5):
                    if idx < n:
                        centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
                        idx += 1
            # Add last one at center? No, center occupied. Add at edge?
            if idx < n:
                centers[idx] = [0.0, 0.5]
                idx += 1
            return centers
            
        else: # random
            for i in range(n):
                centers[i] = [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)]
            return centers

    # Optimization loop
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Try multiple initializations
    for attempt in range(5):
        centers = generate_initial_centers('hex')
        # Small random perturbation
        centers += np.random.normal(0, 0.01, centers.shape)
        # Clip to valid range (away from walls slightly)
        centers = np.clip(centers, 0.01, 0.99)

        current_sum, current_radii = solve_radii(centers)
        
        # Local optimization using repulsion
        # We want to move centers apart where radii are constrained by distance
        step_size = 0.005
        iterations = 200
        
        for it in range(iterations):
            # Solve LP to get current radii
            s, r = solve_radii(centers)
            if s <= best_sum:
                # If not improving globally, maybe continue local search?
                pass
            
            # Identify tight constraints (r_i + r_j approx dist)
            # And compute forces
            forces = np.zeros_like(centers)
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    # If constraint is active (close to equality)
                    if dist > 1e-9:
                        # Slack
                        slack = dist - (r[i] + r[j])
                        # If slack is small positive or negative (due to numerical noise), apply force
                        # We want to increase slack => increase dist => move apart
                        # Force magnitude proportional to how tight?
                        # Or just constant repulsion if tight?
                        if slack < 1e-4: # Active constraint
                            direction = (centers[i] - centers[j]) / dist
                            # Push i away from j, j away from i
                            # Strength? Maybe proportional to radii?
                            f = step_size * 0.1 
                            forces[i] += f * direction
                            forces[j] -= f * direction
            
            # Apply forces
            centers += forces
            
            # Project back to [0.01, 0.99] to avoid wall issues during optimization
            # Actually, boundary constraints are handled by LP, but centers should stay in square
            centers = np.clip(centers, 0.01, 0.99)

        # Final solve for this run
        final_sum, final_radii = solve_radii(centers)
        
        if final_sum > best_sum:
            best_sum = final_sum
            best_centers = centers.copy()
            best_radii = final_radii.copy()
            
        print(f"Attempt {attempt+1}: Sum = {final_sum:.5f}")

    # Return best found
    # Ensure centers are valid (just in case)
    # Clip centers to [0, 1]
    best_centers = np.clip(best_centers, 0.0, 1.0)
    # Re-solve radii one last time to ensure consistency with clipped centers
    final_sum, final_radii = solve_radii(best_centers)
    
    return best_centers, final_radii, float(final_sum)

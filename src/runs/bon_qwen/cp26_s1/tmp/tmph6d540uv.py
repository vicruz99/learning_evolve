import numpy as np
import scipy.optimize as opt
import math

def solve_radii_lp(centers):
    """
    Given fixed centers, solve for radii that maximize sum of radii
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r) => Minimize -sum(r)
    c_obj = np.ones(n)
    
    # Constraints: A_ub @ r <= b_ub
    
    # 1. Boundary constraints
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # These can be written as 1*r_i <= bound
    
    A_ub_list = []
    b_ub_list = []
    
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(x)
        
        # r_i <= 1 - x
        A_ub_list.append(row)
        b_ub_list.append(1.0 - x)
        
        # r_i <= y
        A_ub_list.append(row)
        b_ub_list.append(y)
        
        # r_i <= 1 - y
        A_ub_list.append(row)
        b_ub_list.append(1.0 - y)
        
    # 2. Pairwise non-overlap constraints
    # r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist > 0: # Avoid division by zero or invalid constraints if dist=0
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub_list.append(row)
                b_ub_list.append(dist)
            else:
                # If centers coincide, radii must be 0
                # Add constraint r_i + r_j <= 0 => r_i=0, r_j=0
                # This is handled by r_i >= 0 bounds, but technically:
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub_list.append(row)
                b_ub_list.append(0.0)

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
        else:
            # Fallback: return small radii if LP fails
            return np.zeros(n), 0.0
    except Exception:
        return np.zeros(n), 0.0

def run_packing():
    """
    Main function to run the optimization.
    """
    n = 26
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_sum = -1.0
    
    # Helper to initialize centers
    def init_centers():
        # Strategy: Random initialization with some structure
        # Try to place them somewhat evenly to avoid extreme overlaps initially
        centers = np.random.rand(n, 2)
        return centers

    # Number of restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        # Initialize centers
        # Use a grid-like structure perturbed randomly for better starting point
        # Grid 5x5 is 25 circles. 26th in middle or corner?
        # Let's just use random for robustness, but maybe clip to avoid boundaries
        centers = np.random.uniform(0.1, 0.9, (n, 2))
        
        # Current state
        current_sum = -1.0
        
        # Solve initial radii
        radii, current_sum = solve_radii_lp(centers)
        
        # Local search parameters
        step_size = 0.05
        decay = 0.995
        min_step = 1e-5
        max_iters = 2000
        
        for iteration in range(max_iters):
            if step_size < min_step:
                break
            
            # Pick a random circle to move
            idx = np.random.randint(0, n)
            
            # Generate perturbation
            delta = np.random.normal(0, step_size, 2)
            new_pos = centers[idx] + delta
            
            # Clip to [0, 1]
            new_pos = np.clip(new_pos, 0.0, 1.0)
            
            # Check if move improves sum of radii
            # Optimization: only update center and re-solve LP
            old_pos = centers[idx].copy()
            centers[idx] = new_pos
            
            # Solve LP for new configuration
            new_radii, new_sum = solve_radii_lp(centers)
            
            if new_sum > current_sum:
                # Accept move
                current_sum = new_sum
                best_radii = new_radii.copy()
                best_centers = centers.copy()
                # Maybe increase step size slightly if improving?
                # step_size = min(step_size * 1.01, 0.1) 
            else:
                # Reject move
                centers[idx] = old_pos
            
            # Decay step size
            step_size *= decay
            
            # Occasionally perform a larger random jump to escape local optima
            if iteration % 100 == 0 and iteration > 0:
                # Random restart of one circle
                idx = np.random.randint(0, n)
                centers[idx] = np.random.uniform(0.05, 0.95, 2)
                radii, current_sum = solve_radii_lp(centers)

        if current_sum > best_sum:
            best_sum = current_sum
            best_radii = radii.copy() # Note: radii variable might be stale, recompute
            best_radii, _ = solve_radii_lp(best_centers)

    # Final validation and cleanup
    # Ensure radii are valid
    final_radii, final_sum = solve_radii_lp(best_centers)
    
    # Check for NaNs or issues
    if np.isnan(final_radii).any():
        # Fallback to a simple grid if optimization failed
        # 5x5 grid + 1 in center?
        centers = np.zeros((n, 2))
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < n:
                    centers[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                    idx += 1
        if idx < n:
            centers[idx] = [0.5, 0.5] # Overlap, but solver will fix radii
        final_radii, final_sum = solve_radii_lp(centers)
        best_centers = centers

    return best_centers, final_radii, final_sum

# Run the packing
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print("Centers:\n", centers)
    print("Radii:\n", radii)
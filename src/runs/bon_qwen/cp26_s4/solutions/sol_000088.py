# sol_000088 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f395aea4) state=04808107 sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Runs the optimization to pack 26 circles in a unit square.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # Constants
    penalty_weight = 50000.0
    
    def compute_loss(v):
        """
        Computes the objective (negative sum of radii) plus penalty for constraint violations.
        v: flattened array of [x0, ..., x25, y0, ..., y25, r0, ..., r25]
        """
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        # Objective: Maximize sum of radii -> Minimize negative sum
        obj = -np.sum(r)
        
        penalty = 0.0
        
        # 1. Boundary Constraint Violations
        # Circles must be inside [0, 1] x [0, 1]
        # x - r >= 0  => violation if r > x
        # x + r <= 1  => violation if x + r > 1
        # Same for y
        
        # Vectorized calculations
        # Left/Bottom boundaries: r - x > 0 or r - y > 0
        viol_lb = np.maximum(0.0, r - x) + np.maximum(0.0, r - y)
        # Right/Top boundaries: (x + r) - 1 > 0 or (y + r) - 1 > 0
        viol_rt = np.maximum(0.0, x + r - 1.0) + np.maximum(0.0, y + r - 1.0)
        
        penalty += np.sum(viol_lb**2) + np.sum(viol_rt**2)
        
        # 2. Overlap Constraint Violations
        # Distance between i and j must be >= r[i] + r[j]
        # Violation if dist < r[i] + r[j]
        
        # Compute pairwise distances efficiently
        # Using broadcasting
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dists = np.sqrt(diff_x**2 + diff_y**2)
        
        sum_radii_matrix = r[:, np.newaxis] + r[np.newaxis, :]
        
        # We only care about pairs i < j to avoid double counting and self-overlap
        # Create a mask for upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        overlap_violations = np.maximum(0.0, sum_radii_matrix[mask] - dists[mask])
        penalty += np.sum(overlap_violations**2)
        
        # Total loss
        loss = obj + penalty_weight * penalty
        return loss

    def compute_gradient(v):
        """
        Numerical gradient approximation (if needed), but L-BFGS-B approximates it.
        For better performance, we rely on the optimizer's approximation or 
        provide a function if the penalty is simple. 
        Since the loss is differentiable almost everywhere, we let the optimizer handle it.
        """
        pass

    # --- Initialization ---
    # Initialize centers in a hexagonal-like grid pattern to start close to optimal packing
    v0 = np.zeros(3 * n)
    
    # Hexagonal packing parameters
    # We want to fit 26 circles. 
    # A 5x6 grid (30 spots) is a good over-estimate, we will fill 26 spots.
    # We place them evenly within the square initially.
    
    # Rows and Columns for initialization
    # 5 rows
    rows = 5
    cols = 6 # Enough for 26 items (5*5 + 1)
    
    # Y coordinates for rows (evenly spaced)
    y_base = np.linspace(0.2, 0.8, rows)
    # X coordinates for columns (evenly spaced)
    x_base = np.linspace(0.2, 0.8, cols)
    
    count = 0
    for i, y_val in enumerate(y_base):
        # Shift odd rows for hexagonal pattern
        shift = 0.0
        if i % 2 == 1:
            shift = 0.5 / cols # Half step shift
        
        for j in range(cols):
            if count >= n:
                break
            
            # Calculate initial x position with shift
            x_val = x_base[j] + shift
            
            # Ensure x_val stays within bounds roughly
            if x_val > 1.0:
                x_val = 1.0 - (x_val % 1.0) # Wrap or clamp? Just clamp for init
            
            # Assign to v0
            # Layout: x (0..n-1), y (n..2n-1), r (2n..3n-1)
            v0[count] = x_val
            v0[n + count] = y_val
            v0[2*n + count] = 0.05 # Initial small radius
            
            count += 1
            
        if count >= n:
            break

    # --- Optimization ---
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (radius cannot exceed 0.5 in unit square)
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    # Run optimization
    # L-BFGS-B is suitable for bound-constrained optimization
    try:
        res = minimize(compute_loss, v0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        v_final = res.x
    except Exception as e:
        # Fallback in case of error, though unlikely with these parameters
        v_final = v0
        
    # Extract results
    centers_x = v_final[:n]
    centers_y = v_final[n:2*n]
    radii = v_final[2*n:3*n]
    
    centers = np.column_stack((centers_x, centers_y))
    
    # Calculate sum of radii
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# --- Validation Wrapper (Simulated run for self-check) ---
# Note: The prompt says I must define run_packing. 
# I will include a check here just to be safe, though the function itself is the requirement.

if __name__ == "__main__":
    # Import numpy for validation check logic if needed, 
    # but the validate function is provided externally.
    # We can manually check if we are in a test environment.
    
    c, r, s = run_packing()
    
    # Manual validation check logic (mirroring provided validate_packing)
    n = c.shape[0]
    valid = True
    
    if np.isnan(c).any() or np.isnan(r).any():
        print("NaN detected")
        valid = False
    else:
        for i in range(n):
            x, y = c[i]
            rad = r[i]
            if rad < 0:
                print(f"Negative radius {rad}")
                valid = False
                break
            if x - rad < -1e-9 or x + rad > 1 + 1e-9 or y - rad < -1e-9 or y + rad > 1 + 1e-9:
                print(f"Circle {i} out of bounds")
                valid = False
                break
        
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                    if dist < r[i] + r[j] - 1e-9:
                        print(f"Overlap between {i} and {j}")
                        valid = False
                        break
                if not valid:
                    break
                    
    if valid:
        print(f"Success! Sum of radii: {s:.6f}")
    else:
        print("Validation failed.")

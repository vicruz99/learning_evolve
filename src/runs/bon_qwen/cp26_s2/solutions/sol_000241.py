# sol_000241 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6efaf445) state=0f54d9c3 sum of radii=0.260000 correctness=1.0
# stdout(first 200): Circle 0 at (0.05, 0.05) with radius 1.0 is outside the unit square Result invalid. Attempting repair... Circle 0 at (0.05, 0.05) with radius 0.99 is outside the unit square Circle 0 at (0.05, 0.05) w
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def generate_initial_positions(n, side=1.0):
    """
    Generates initial positions for n circles in a hexagonal pattern.
    """
    positions = []
    # Estimate radius for initialization
    # Area per circle approx 1/n. r approx sqrt(1/(pi*n))
    # But for packing, we need smaller. Let's start with r=0.05
    r_init = 0.05
    
    # Hexagonal packing parameters
    dx = 2 * r_init
    dy = r_init * math.sqrt(3)
    
    x = r_init
    y = r_init
    
    row_idx = 0
    while len(positions) < n:
        row_start = r_init if row_idx % 2 == 0 else r_init + dx/2
        # Actually standard hex packing: even rows at x=r, odd rows at x=r+dx/2
        # But let's just fill rows
        cx = r_init + (row_idx % 2) * (dx / 2)
        
        while cx < side - r_init and len(positions) < n:
            positions.append([cx, y])
            cx += dx
        
        y += dy
        row_idx += 1
        
        # If we exceeded y boundary, stop (though loop condition handles n)
        if y > side - r_init:
            break
            
    # Pad with random positions if needed (should not happen with logic above for n=26)
    while len(positions) < n:
        positions.append([np.random.rand(), np.random.rand()])
        
    return np.array(positions[:n])

def run_packing():
    n = 26
    
    # 1. Initialize positions and radii
    # Using a slightly larger radius for initialization to push boundaries
    # But solver will adjust. Let's use a reasonable grid/hex.
    init_positions = generate_initial_positions(n)
    # Start with radii slightly smaller than what we hope for, e.g., 0.08
    init_radii = np.full(n, 0.08)
    
    # Flatten variables for optimization: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.concatenate([init_positions.flatten(), init_radii])
    
    # Define constraints
    # We need:
    # 1. x_i >= r_i  => x_i - r_i >= 0
    # 2. x_i <= 1 - r_i => 1 - x_i - r_i >= 0
    # 3. y_i >= r_i => y_i - r_i >= 0
    # 4. y_i <= 1 - r_i => 1 - y_i - r_i >= 0
    # 5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # This is a lot of constraints for scipy. 
    # Instead, let's use a penalty method or a simplified optimizer.
    # Or use SLSQP with constraints.
    
    # Let's try SLSQP.
    # Number of variables: 3 * 26 = 78.
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]
        })
        # y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]
        })

    # Overlap constraints
    # Only for pairs that are likely to overlap or all pairs?
    # All pairs is O(n^2) = 325 constraints. Feasible.
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = 3 * i
            idx_yi = 3 * i + 1
            idx_ri = 3 * i + 2
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            def dist_sq_minus_sum_r_sq(v, i=i, j=j):
                xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
                xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                return dist_sq - sum_r**2
            
            constraints.append({
                'type': 'ineq',
                'fun': dist_sq_minus_sum_r_sq
            })

    # Objective: Maximize sum of radii => Minimize -sum(radii)
    def objective(v):
        radii = v[2::3]
        return -np.sum(radii)

    # Bounds for variables
    # x, y in [0, 1]
    # r >= 0
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 1)) # r (upper bound 1 is safe)

    # Run optimization
    # SLSQP can be slow with many constraints. 
    # We might need to be careful.
    # Let's try to run it.
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
        
        if res.success:
            final_v = res.x
        else:
            # If failed, try to repair or return best found
            final_v = x0 
            # Maybe run a few more iterations or use the result anyway
            # Check if result is feasible?
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        final_v = x0

    # Extract centers and radii
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = final_v[3*i]
        centers[i, 1] = final_v[3*i + 1]
        radii[i] = final_v[3*i + 2]
        
    # Clip radii to be non-negative (numerical safety)
    radii = np.maximum(radii, 0)
    
    # Validate
    if not validate_packing(centers, radii):
        print("Result invalid. Attempting repair...")
        # Simple repair: shrink radii slightly until valid
        scale = 0.99
        while scale > 0.1:
            test_radii = radii * scale
            # Centers might need adjustment?
            # If radii shrink, centers are still valid for boundary if they were.
            # Overlaps might resolve.
            # But boundary constraints: if center was close to edge, shrinking r helps.
            # So just shrinking radii usually fixes overlap and boundary issues.
            if validate_packing(centers, test_radii):
                radii = test_radii
                break
            scale *= 0.95
        else:
            # If still invalid, force very small radii
            radii = np.full(n, 0.01)
            centers = np.random.rand(n, 2) * 0.98 + 0.01 # Random small placement
            
            # Re-run quick optimization?
            # For now, return a valid small packing
            pass

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

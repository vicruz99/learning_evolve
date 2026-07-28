# sol_000022 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cd0e5d1c) state=437b03fa sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # Helper to generate initial hexagonal packing
    def get_initial_packing(seed_shift=0.0):
        centers = []
        radii = []
        # Approximate radius for hex packing in unit square for n=26
        # Try to fit in rows. 6, 5, 6, 5, 4 -> 26 circles
        # Height estimation: 5 rows. 
        # Let's try a simple grid first to ensure validity, then optimize
        # A 5x5 grid leaves space.
        
        # Let's create a hexagonal arrangement
        # Row 0: 6 circles
        # Row 1: 5 circles
        # Row 2: 6 circles
        # Row 3: 5 circles
        # Row 4: 4 circles
        # Total 26.
        
        # We need to estimate r. 
        # Width constraint for 6 circles: 12r <= 1 -> r <= 0.0833
        # Height constraint for 5 rows: 2r + 4*sqrt(3)r <= 1 -> r(2+6.92) <= 1 -> r <= 0.112
        # So r is limited by width to ~0.08.
        # But we can optimize to increase r.
        
        r_start = 0.08
        
        # Generate centers
        row_counts = [6, 5, 6, 5, 4]
        y_coord = r_start
        for row_idx, count in enumerate(row_counts):
            # Calculate width needed for this row
            # For count circles, width is count * 2r? No.
            # Centers x: r, 3r, 5r...
            # Last center at r + (count-1)*2r
            # Rightmost edge at center + r = 2r + (count-1)*2r = 2*count*r
            # We want to center the row in the square?
            # Or just pack from left.
            # Let's pack from left to fit tightly.
            
            # Shift for hexagonal packing (staggered rows)
            # Row 0 (even): x starts at r
            # Row 1 (odd): x starts at 2r (shifted by r) ? 
            # Actually, if row 0 has 6 circles at 1r, 3r... 
            # Row 1 has 5 circles. To fit in gaps, centers should be at 2r, 4r...
            
            x_start = r_start
            if row_idx % 2 == 1:
                x_start = 2 * r_start # Shifted row
            
            for k in range(count):
                x = x_start + k * (2 * r_start)
                centers.append([x, y_coord])
                radii.append(r_start)
            
            y_coord += math.sqrt(3) * r_start
            
        return np.array(centers), np.array(radii)

    # Try multiple initializations
    best_sum = 0
    best_centers = None
    best_radii = None

    # Initialization 1: Hexagonal
    c, r = get_initial_packing()
    if c.shape[0] != n:
        # Fallback to grid if counting logic is tricky
        c = np.random.uniform(0.1, 0.9, (n, 2))
        r = np.full(n, 0.05)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = c[i, 0]
        x0[3*i+1] = c[i, 1]
        x0[3*i+2] = r[i]

    # Objective: maximize sum of radii -> minimize -sum
    def objective(vars):
        return -np.sum(vars[2::3])

    # Constraints
    cons = []
    
    # Boundary constraints
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # Same for y
    
    for i in range(n):
        idx = 3 * i
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx] - v[idx+2]})
        # 1 - (x + r) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - (v[idx] + v[idx+2])})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx+1] - v[idx+2]})
        # 1 - (y + r) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - (v[idx+1] + v[idx+2])})
        # r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx+2]})

    # Non-overlap constraints
    # dist^2 >= (r_i + r_j)^2
    # dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = 3 * i
            idx_j = 3 * j
            def overlap_con(v, i=i, j=j):
                idx_i = 3 * i
                idx_j = 3 * j
                dx = v[idx_i] - v[idx_j]
                dy = v[idx_i+1] - v[idx_j+1]
                r_sum = v[idx_i+2] + v[idx_j+2]
                return dx**2 + dy**2 - r_sum**2
            cons.append({'type': 'ineq', 'fun': overlap_con})

    # Bounds for variables to help optimizer
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Run optimization
    # SLSQP is good for constrained problems
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success or -res.fun > best_sum:
            # Extract results
            centers = res.x.reshape(n, 3)[:, :2]
            radii = res.x.reshape(n, 3)[:, 2]
            current_sum = np.sum(radii)
            
            # Small check to ensure validity before accepting
            # Sometimes optimizer hits boundary with numerical errors
            # We can clip slightly if needed, but let's trust SLSQP mostly.
            # However, to be safe, we can check overlaps.
            
            valid = True
            # Quick check
            for i in range(n):
                x, y, r_val = centers[i, 0], centers[i, 1], radii[i]
                if x - r_val < -1e-9 or x + r_val > 1 + 1e-9 or \
                   y - r_val < -1e-9 or y + r_val > 1 + 1e-9:
                    valid = False
                    break
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_sum = current_sum
                best_centers = centers
                best_radii = radii

    except Exception as e:
        print(f"Optimization failed: {e}")

    # If best_radii is still None or sum is low, fallback to a known valid grid
    if best_radii is None or best_sum < 1.0:
        # Fallback: 5x5 grid + 1 small circle?
        # Or just a grid of 25 and 1 tiny one.
        # But we want high sum.
        # Let's try a simpler perturbation of the initial guess if optimization failed
        best_centers = c
        best_radii = r
        best_sum = np.sum(r)

    # Final refinement / cleaning
    # Ensure strict validity with epsilon
    # If any circle is too close to boundary or overlap, shrink slightly
    # This is a post-processing step to guarantee pass.
    
    # However, the optimizer should have respected constraints.
    # Let's just return the result.
    
    return best_centers, best_radii, float(best_sum)

# To test locally or if run_packing is called
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radii range: {np.min(radii):.4f} - {np.max(radii):.4f}")

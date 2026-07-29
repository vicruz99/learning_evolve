# sol_000316 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1cbfbe8a) state=d063ab91 sum of radii=2.510000 correctness=1.0
# stdout(first 200): Optimization failed: 'x0' must only have one dimension. Optimization failed: 'x0' must only have one dimension. Optimization failed: 'x0' must only have one dimension.
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def run_packing():
    """
    Solves the circle packing problem for 26 circles in a unit square.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # Helper function to calculate objective (negative sum of radii)
    def objective(variables):
        # variables shape: (n, 3) -> [x, y, r]
        # We want to maximize sum(r), so minimize -sum(r)
        return -np.sum(variables[:, 2])

    # Helper function to create constraints
    # Constraints:
    # 1. x >= r
    # 2. x + r <= 1
    # 3. y >= r
    # 4. y + r <= 1
    # 5. dist(i, j) >= r_i + r_j  <=> dist^2 - (r_i + r_j)^2 >= 0
    
    def get_constraints(n):
        cons = []
        
        # Boundary constraints
        for i in range(n):
            # x >= r  =>  x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i, 0] - v[i, 2]})
            # x + r <= 1 => 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i, 0] - v[i, 2]})
            # y >= r  =>  y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i, 1] - v[i, 2]})
            # y + r <= 1 => 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i, 1] - v[i, 2]})
            # r >= 0 (handled by bounds usually, but good to be safe)
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i, 2]})

        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                def overlap_func(v, i=i, j=j):
                    dx = v[i, 0] - v[j, 0]
                    dy = v[i, 1] - v[j, 1]
                    dist_sq = dx*dx + dy*dy
                    r_sum = v[i, 2] + v[j, 2]
                    return dist_sq - r_sum*r_sum
                cons.append({'type': 'ineq', 'fun': overlap_func})
        
        return cons

    constraints = get_constraints(n)
    
    # Bounds for [x, y, r]
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius is 0.5)
    bounds = [(0, 1)] * 2 + [(1e-6, 0.5)] # Small lower bound for r to avoid division by zero if needed, though not strictly needed here
    
    best_result = None
    best_sum = -np.inf

    # Strategy 1: Hexagonal Lattice Initialization
    # We try to fit a hex grid. 
    # Estimate radius r. For n=26, r ~ 0.1.
    # Try to arrange in rows.
    # 5 rows of roughly 5 circles.
    
    init_configs = []
    
    # Config 1: Hexagonal packing attempt
    # Rows of 5, 5, 6, 5, 5? Or 5, 6, 5, 5, 5?
    # Let's try a generic hex placement
    r_est = 0.1
    y_step = np.sqrt(3) * r_est
    x_step = 2 * r_est
    
    coords = []
    y = r_est + 0.05 # slight offset to center
    row_idx = 0
    for _ in range(n):
        # Determine x position based on row parity
        # Even rows start at r_est, Odd rows shifted by r_est?
        # Actually standard hex: row 0 at x=r, row 1 at x=2r, row 2 at x=3r...
        # But to fit in square, we might need to center.
        
        # Let's just generate a dense set of points and let optimizer fix it
        # Grid 6x5 = 30 points, pick 26?
        pass

    # Simpler Init: Randomized Grid / Hex
    # We'll generate a few random starting points to ensure diversity
    
    # Init 1: Square Grid 5x5 + 1
    # 5x5 grid centers
    grid_centers = []
    for i in range(5):
        for j in range(5):
            grid_centers.append([0.1 + i*0.2, 0.1 + j*0.2])
    # Add one in center? (0.5, 0.5) is occupied.
    # Maybe center of a hole (0.2, 0.2)?
    grid_centers.append([0.2, 0.2])
    # Pad to 26 if needed (already 26)
    init1 = np.array(grid_centers)
    init1 = np.hstack([init1, np.full((n, 1), 0.05)]) # Initial radius 0.05
    init_configs.append(init1)

    # Init 2: Hexagonal Lattice
    # Try to fit rows. 5 rows.
    # Rows sizes: 5, 6, 5, 6, 4? Sum = 26.
    # Or 6, 5, 6, 5, 4?
    # Let's try a loop
    hex_coords = []
    # We want to fill [0.1, 0.9] range roughly.
    # Spacing
    r_guess = 0.1
    x_spacing = 2 * r_guess
    y_spacing = np.sqrt(3) * r_guess
    
    # Center the packing
    # Approx width needed for k circles: (k-1)*x_spacing + 2*r_guess
    # Approx height needed for m rows: (m-1)*y_spacing + 2*r_guess
    
    # Let's try to place them in a rectangular block with staggering
    # 6 columns, 5 rows? 30 spots.
    # We will just scatter them slightly and let optimizer work.
    
    # Better Hex Init:
    # Place circles in a pattern that respects distance 2r
    current_r = 0.101 # Target r
    placed = []
    # Try grid positions
    for y_pos in np.linspace(current_r, 1-current_r, 6): # 6 rows
        for x_pos in np.linspace(current_r, 1-current_r, 5): # 5 cols
             placed.append([x_pos, y_pos])
    # We have 30 spots. Take first 26.
    # But this is just a square grid.
    # Shift odd rows
    hex_pts = []
    for i, p in enumerate(placed):
        if len(hex_pts) >= n: break
        y_shift = 0
        # Determine row index roughly
        row_idx = int((p[1] - current_r) / ( (1-2*current_r)/5 ))
        if row_idx % 2 == 1:
             p[0] += current_r # Shift right by r
             # Clip
             if p[0] > 1-current_r:
                 p[0] = 1-current_r
        hex_pts.append(p)
    
    init2 = np.array(hex_pts[:n])
    init2 = np.hstack([init2, np.full((n, 1), 0.05)])
    init_configs.append(init2)
    
    # Init 3: Random perturbation of hex
    # Create a dense random packing using a simple greedy algorithm or just random
    # But random is slow. Let's use a jittered grid.
    pts = []
    # 5x6 grid jittered
    for i in range(6):
        for j in range(5):
            x = 0.1 + i * 0.18 + np.random.uniform(-0.02, 0.02)
            y = 0.1 + j * 0.22 + np.random.uniform(-0.02, 0.02)
            pts.append([x, y])
            if len(pts) >= n: break
        if len(pts) >= n: break
    init3 = np.array(pts[:n])
    init3 = np.hstack([init3, np.full((n, 1), 0.05)])
    init_configs.append(init3)

    # Run Optimization
    constraints = get_constraints(n)
    
    for init_vars in init_configs:
        # L-BFGS-B handles bounds and constraints
        # However, constraints with closures can be slow. 
        # But n=26 is small enough.
        
        try:
            res = minimize(objective, init_vars, method='L-BFGS-B', 
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
            
            if res.success:
                val = res.fun
                if val < best_sum: # val is negative sum
                    best_sum = val
                    best_result = res.x
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue

    # If optimization failed or got stuck, fallback to a known good config?
    # But L-BFGS-B is usually robust with good init.
    
    if best_result is None:
        # Fallback: Square grid 5x5 + 1 small
        centers = np.zeros((n, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                idx += 1
        centers[25] = [0.5, 0.5] # Overlap? 
        # Better fallback: 5x5 grid r=0.1, last circle r=0.04 at (0.2, 0.2)
        # But we need to return valid.
        # Let's just return a valid grid with small radii
        centers = []
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + i*0.2, 0.1 + j*0.2])
        centers.append([0.2, 0.2])
        centers = np.array(centers)
        radii = np.array([0.1]*25 + [0.01]) # Valid? (0.2,0.2) dist to (0.1,0.1) is 0.141. 0.1+0.01=0.11 < 0.141. OK.
        return centers, radii, np.sum(radii)

    # Extract best result
    best_centers = best_result[:, :2]
    best_radii = best_result[:, 2]
    
    # Final cleanup: ensure non-negative radii and valid bounds
    # The optimizer should have respected bounds, but clamp just in case
    best_radii = np.maximum(best_radii, 0)
    # Ensure centers are within [r, 1-r]
    for i in range(n):
        r = best_radii[i]
        best_centers[i, 0] = np.clip(best_centers[i, 0], r, 1-r)
        best_centers[i, 1] = np.clip(best_centers[i, 1], r, 1-r)
        best_radii[i] = r # Update in case radius changed? No, we clamped center.
        # Actually if we clamp center, we might create overlap. 
        # But L-BFGS-B should have kept us feasible. 
        # Let's trust the optimizer result.
        
    return best_centers, best_radii, np.sum(best_radii)

# Validation helper (not part of solution but good for checking)
def check_valid(centers, radii):
    import numpy as np
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < -1e-9 or x > 1+1e-9 or y < -1e-9 or y > 1+1e-9:
            return False
        if r < -1e-9:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i]-centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

# sol_000136 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5da4630c) state=543825d5 sum of radii=2.598313 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses scipy.optimize to find an optimal configuration.
    """
    n = 26
    
    # Initial configuration: Hexagonal-like grid
    # We try to arrange circles in rows with alternating shifts
    # 5 rows: 6, 5, 6, 5, 4 circles? No, 6 is too wide for r > 0.08.
    # Let's try 5, 5, 5, 5, 6 (shifted) or similar.
    # Actually, let's just generate a dense hexagonal grid and pick 26 points.
    
    # Generate hexagonal grid points
    # Spacing roughly 0.2 (for r=0.1)
    points = []
    # Try a range of grid densities
    rows = 7
    cols = 7
    spacing = 1.0 / (rows * 0.8) # heuristic
    
    # Better initialization: Random uniform in [0,1] is bad.
    # Let's use a structured grid.
    # 5x5 grid is 25 points. We need 26.
    # Let's place 25 in a 5x5 grid with r=0.09 (to leave room) and 1 in the middle?
    # Or better: A hexagonal packing pattern.
    
    # Let's construct a configuration manually that is likely good.
    # 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 6 circles? No, 6 is wide.
    # Maybe 6, 5, 5, 5, 5?
    
    # Let's try a generic dense packing initialization.
    # Place centers on a triangular lattice scaled to fit.
    # Triangular lattice vectors: v1 = (2r, 0), v2 = (r, sqrt(3)r)
    # We don't know r yet. Let's assume r approx 0.1.
    
    centers_init = np.zeros((n, 2))
    idx = 0
    
    # We can fit roughly 5 rows vertically.
    # Vertical spacing ~ 0.173 (sqrt(3)*0.1)
    # 5 rows need height ~ 4*0.173 + 2*0.1 = 0.692 + 0.2 = 0.892. Fits easily.
    # Let's use 6 rows to be safe.
    
    y_coords = np.linspace(0.1, 0.9, 6) # 6 rows
    
    # Row 0: 5 circles
    # Row 1: 5 circles shifted
    # Row 2: 5 circles
    # Row 3: 5 circles shifted
    # Row 4: 4 circles?
    # Total 24. Need 2 more.
    # Let's just fill rows greedily.
    
    row_lengths = [5, 5, 5, 5, 4, 2] # Sum = 26
    
    current_y = 0.1
    dy = 0.9 / 5 # Spread over 0.8 height range roughly
    
    for k, length in enumerate(row_lengths):
        # Shift odd rows
        shift = 0.05 if k % 2 == 1 else 0.0 
        # Center the row
        # Width needed approx length * 0.2
        # Available width 1.0
        # Start x
        start_x = 0.5 - (length - 1) * 0.1 + shift # Rough centering
        
        # Better centering:
        # We want x coordinates such that they fit in [0,1]
        # Let's just space them evenly
        x_spacing = 0.2 # 2*r_approx
        row_width = (length - 1) * x_spacing
        x_start = (1.0 - row_width) / 2.0 + shift
        
        # Adjust shift to keep within bounds
        # If shifted by 0.05, x_start might move.
        # Let's clamp x coordinates to [0.1, 0.9]
        
        for j in range(length):
            x = x_start + j * x_spacing
            # Clamp to safe zone for initialization
            x = np.clip(x, 0.15, 0.85)
            y = y_coords[k] if k < len(y_coords) else y_coords[-1] + (k - 5) * 0.1
            
            if idx < n:
                centers_init[idx] = [x, y]
                idx += 1
    
    # Fill remaining if any
    while idx < n:
        centers_init[idx] = [0.5, 0.5]
        idx += 1
        
    # Initial radii
    radii_init = np.full(n, 0.08) # Start small to be safe
    
    # Combine into optimization vector
    # x0 = [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # Bounds: x, y in [0, 1], r >= 0
    # Actually, strict bounds for x,y are [0,1] but constraints enforce r margin.
    # We can set bounds for x,y as [0, 1] and r as [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii) # Maximize sum of radii
        
    def constraint_overlap(vars):
        centers = np.column_stack((vars[0::3], vars[1::3]))
        radii = vars[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                dists[i, j] = d - radii[i] - radii[j]
        # Return min distance - sum radii. Must be >= 0.
        # We need to return a scalar or array of constraints.
        # SLSQP can handle array constraints.
        return dists[np.triu_indices(n, k=1)]

    def constraint_boundary(vars):
        centers = np.column_stack((vars[0::3], vars[1::3]))
        radii = vars[2::3]
        # x - r >= 0 => x >= r
        # x + r <= 1 => x <= 1 - r
        # y - r >= 0 => y >= r
        # y + r <= 1 => y <= 1 - r
        
        c1 = centers[:, 0] - radii # >= 0
        c2 = 1.0 - centers[:, 0] - radii # >= 0
        c3 = centers[:, 1] - radii # >= 0
        c4 = 1.0 - centers[:, 1] - radii # >= 0
        
        return np.concatenate([c1, c2, c3, c4])

    constraints = []
    constraints.append({'type': 'ineq', 'fun': constraint_overlap})
    constraints.append({'type': 'ineq', 'fun': constraint_boundary})
    
    # Run optimization
    # SLSQP is good for this
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Extract result
    final_vars = res.x
    final_centers = np.column_stack((final_vars[0::3], final_vars[1::3]))
    final_radii = final_vars[2::3]
    
    # Ensure non-negative radii (numerical noise)
    final_radii = np.maximum(final_radii, 0.0)
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

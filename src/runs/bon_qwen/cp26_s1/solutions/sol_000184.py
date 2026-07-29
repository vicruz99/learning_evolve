# sol_000184 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5c6e3651) state=1fbe39cf sum of radii=2.607072 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square.
    """
    n = 26
    
    # --- 1. Initialization: Hexagonal Grid Pattern ---
    # We aim to distribute 26 points in a square using a hexagonal lattice logic.
    # A pattern like 5-4-5-4-5-3 (sum=26) works well for hexagonal packing in a square.
    
    centers_init = []
    
    # Define row parameters
    # We will construct rows with alternating x-offsets to mimic hexagonal packing.
    # Vertical spacing for hexagonal packing is r * sqrt(3).
    # Horizontal spacing is 2 * r.
    # Since we don't know optimal r yet, we use relative spacing.
    
    # Let's define a target layout:
    # Row 0: 5 circles
    # Row 1: 4 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 4 circles (shifted)
    # Row 4: 5 circles
    # Row 5: 3 circles (centered)
    
    # Approximate vertical positions to fill [0, 1]
    # 6 rows means 5 gaps. 
    # We can place rows at y = 0.1, 0.26, 0.42, 0.58, 0.74, 0.9?
    # Let's just use linspace for now, optimizer will fix it.
    
    # Better approach: Generate points on a grid and let optimizer fix overlaps.
    # Let's create a dense set of points.
    
    # Simple Grid Initialization
    # 6 columns, 5 rows = 30 points. We only need 26.
    # Let's pick the "best" 26 from a 6x5 grid or just place them.
    
    # Let's try to place them manually to look like a hex lattice
    # y-coords for 6 rows
    y_coords = np.linspace(0.1, 0.9, 6) 
    
    current_centers = []
    
    # Row 0: 5 points
    x_row = np.linspace(0.1, 0.9, 5)
    for x in x_row:
        current_centers.append([x, y_coords[0]])
        
    # Row 1: 4 points (shifted by half width)
    # If 5 points span 0.1 to 0.9 (width 0.8), spacing 0.2.
    # Shift by 0.1. Range 0.2 to 0.8.
    x_row = np.linspace(0.2, 0.8, 4)
    for x in x_row:
        current_centers.append([x, y_coords[1]])
        
    # Row 2: 5 points (aligned with Row 0)
    x_row = np.linspace(0.1, 0.9, 5)
    for x in x_row:
        current_centers.append([x, y_coords[2]])
        
    # Row 3: 4 points (shifted)
    x_row = np.linspace(0.2, 0.8, 4)
    for x in x_row:
        current_centers.append([x, y_coords[3]])
        
    # Row 4: 5 points
    x_row = np.linspace(0.1, 0.9, 5)
    for x in x_row:
        current_centers.append([x, y_coords[4]])
        
    # Row 5: 3 points (centered)
    # Need 3 points. Center at 0.5. Spacing 0.2?
    # 0.3, 0.5, 0.7
    x_row = np.linspace(0.3, 0.7, 3)
    for x in x_row:
        current_centers.append([x, y_coords[5]])
        
    centers = np.array(current_centers)
    
    # Initial radii: small enough to not overlap
    # Estimate min distance between points
    min_dist = 1.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    
    # Start with radii slightly smaller than half min distance
    r_init = min_dist * 0.4 
    radii = np.full(n, r_init)
    
    # --- 2. Optimization Setup ---
    
    # Variables: [x0, y0, r0, x1, y1, r1, ...]
    # Total 3 * n variables
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds
    # x, y in [0, 1], r in [0, 1] (tighter bound for r could be 0.5)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)]) # r <= 0.5
        
    # Constraints
    constraints = []
    
    # 1. Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
    # x - r >= 0  => -x + r <= 0
    # 1 - x - r >= 0 => x + r <= 1
    # Same for y
    
    for i in range(n):
        idx_x = 3*i
        idx_y = 3*i + 1
        idx_r = 3*i + 2
        
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]
        })

    # 2. Pairwise non-overlap constraints
    # dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # To avoid sqrt and ensure differentiability, we can use squared distance?
    # But ri + rj is linear.
    # Sqrt is differentiable almost everywhere.
    # Let's use the form: dist - (ri + rj) >= 0
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3*i, 3*i+1, 3*i+2
            idx_xj, idx_yj, idx_rj = 3*j, 3*j+1, 3*j+2
            
            # Capture loop variables
            def make_constraint(i, j):
                def fun(v):
                    xi, yi = v[3*i], v[3*i+1]
                    xj, yj = v[3*j], v[3*j+1]
                    ri, rj = v[3*i+2], v[3*j+2]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - (ri + rj)
                return fun
            
            constraints.append({
                'type': 'ineq',
                'fun': make_constraint(i, j)
            })

    # Objective function: Maximize sum(r) => Minimize -sum(r)
    def objective(v):
        total_r = 0.0
        for i in range(n):
            total_r += v[3*i+2]
        return -total_r

    # --- 3. Run Optimizer ---
    # SLSQP is a good choice for constrained nonlinear optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
    )
    
    # --- 4. Extract Results ---
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = res.x[3*i]
        final_centers[i, 1] = res.x[3*i+1]
        final_radii[i] = res.x[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    # Basic sanity check and clipping if necessary (due to numerical noise)
    # Although constraints should handle it, let's ensure non-negative
    final_radii = np.maximum(final_radii, 0.0)
    
    # Ensure centers are within bounds relative to radius
    for i in range(n):
        r = final_radii[i]
        x = final_centers[i, 0]
        y = final_centers[i, 1]
        
        # Clamp center if needed (though constraints should prevent this)
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        final_centers[i] = [x, y]

    return final_centers, final_radii, sum_radii

# Helper to run if called directly (though instructions say define run_packing)
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(centers)
    # print(radii)

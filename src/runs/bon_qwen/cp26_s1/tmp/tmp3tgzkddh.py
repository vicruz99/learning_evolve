import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Initialization ---
    # Generate a hexagonal grid to initialize positions
    # We create a grid with spacing small enough to cover the square densely
    # then pick 26 points.
    
    points = []
    # Hexagonal grid generation
    # Row spacing = sqrt(3)/2 * spacing
    # Col spacing = spacing
    spacing = 0.15 # Initial spacing (approx diameter 0.3)
    
    y = 0.1 # Start a bit inside to allow room
    row = 0
    while y <= 1.0:
        x = 0.1
        col = 0
        # Offset every other row for hex packing
        offset = spacing / 2.0 if row % 2 == 1 else 0.0
        x = 0.1 + offset
        
        while x <= 1.0:
            points.append([x, y])
            x += spacing
            col += 1
        y += spacing * np.sqrt(3) / 2.0
        row += 1
        
    # If we have fewer than 26 points, generate more or just pad (unlikely with spacing 0.15)
    # If we have more, we just take the first 26. 
    # To make it robust, let's shuffle or just take first. 
    # Taking first 26 from a grid starting at corner is fine.
    
    if len(points) < n:
        # Fallback to random if grid didn't yield enough (shouldn't happen with 0.15)
        np.random.seed(42)
        points = np.random.rand(n, 2)
    else:
        points = points[:n]
        
    centers_init = np.array(points)
    radii_init = np.full(n, 0.05) # Small initial radius
    
    # --- Optimization ---
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    x0 = np.hstack([centers_init, radii_init])
    
    def objective(vars):
        # Maximize sum of radii => Minimize negative sum
        radii = vars[2::3]
        return -np.sum(radii)
    
    # Constraints
    constraints = []
    
    # Boundary constraints for each circle
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: vars[idx_x] - vars[idx_r]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: 1.0 - vars[idx_x] - vars[idx_r]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: vars[idx_y] - vars[idx_r]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: 1.0 - vars[idx_y] - vars[idx_r]
        })
        
    # Non-overlap constraints for each pair
    # dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
    
    # To speed up, we can use a helper or just list comprehension.
    # Note: lambda captures by reference, so we need to bind i, j.
    
    # We can precompute indices to avoid overhead in lambda creation if needed,
    # but for n=26, 325 constraints is manageable.
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3 * i, 3 * i + 1, 3 * i + 2
            idx_xj, idx_yj, idx_rj = 3 * j, 3 * j + 1, 3 * j + 2
            
            def dist_constraint(vars, i=i, j=j, 
                                ix=idx_xi, iy=idx_yi, ir=idx_ri,
                                jx=idx_xj, jy=idx_yj, jr=idx_rj):
                dx = vars[ix] - vars[jx]
                dy = vars[iy] - vars[jy]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - (vars[ir] + vars[jr])
            
            constraints.append({
                'type': 'ineq',
                'fun': dist_constraint
            })
            
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Run optimization
    # SLSQP is good for constrained optimization
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        vars_opt = res.x
    except Exception as e:
        # Fallback if optimization fails
        vars_opt = x0
        
    # Extract results
    centers = np.column_stack((vars_opt[0::3], vars_opt[1::3]))
    radii = vars_opt[2::3]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(f"Centers:\n{c}")
    # print(f"Radii:\n{r}")
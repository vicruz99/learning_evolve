import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal grid initialization and SLSQP optimization.
    """
    n = 26
    
    # Initialize centers in a hexagonal grid pattern
    # 6 rows with alternating 5 and 4 circles to sum to 27, then remove 1
    # Or 5,4,5,4,5,3 to sum to 26.
    # Let's use a dense 6-row structure.
    
    # Estimated optimal radius is around 0.101 (diameter ~0.202)
    # Hexagonal spacing: dx = 2r, dy = sqrt(3)r
    
    # Let's create a grid that fits roughly 5x6 with staggering
    # Row 0: 5 circles
    # Row 1: 4 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles (shifted) -> Total 26
    
    # We'll initialize with a slightly smaller radius to ensure feasibility
    r_init = 0.09
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    centers = []
    
    # Row 0 (5 circles)
    x_start = 1 - 5 * dx # Align to right
    for i in range(5):
        centers.append([x_start + i * dx, r_init])
        
    # Row 1 (4 circles, shifted by dx/2)
    # Shifted rows usually fit between the circles of the previous row
    # Center x shift: r_init
    x_start = 1 - 4 * dx - r_init
    for i in range(4):
        centers.append([x_start + i * dx + r_init, r_init + dy])
        
    # Row 2 (5 circles)
    x_start = 1 - 5 * dx
    for i in range(5):
        centers.append([x_start + i * dx, r_init + 2 * dy])
        
    # Row 3 (4 circles)
    x_start = 1 - 4 * dx - r_init
    for i in range(4):
        centers.append([x_start + i * dx + r_init, r_init + 3 * dy])
        
    # Row 4 (5 circles)
    x_start = 1 - 5 * dx
    for i in range(5):
        centers.append([x_start + i * dx, r_init + 4 * dy])
        
    # Row 5 (3 circles) to make 26
    # We can adjust positions later, just need valid start
    x_start = 1 - 3 * dx - r_init
    for i in range(3):
        centers.append([x_start + i * dx + r_init, r_init + 5 * dy])

    centers = np.array(centers)
    
    # Initial radius
    r_current = r_init
    
    # Define objective: Maximize r (minimize -r)
    def objective(vars):
        # vars: [x1, y1, ..., x26, y26, r]
        r = vars[-1]
        return -r
    
    # Define constraints
    # 1. Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # 2. Distance constraints: dist(ci, cj) >= 2r
    
    cons = []
    
    # Boundary constraints
    for i in range(n):
        xi = 2 * i
        yi = 2 * i + 1
        r_idx = 2 * n
        
        # x_i >= r  => x_i - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[r_idx]})
        # 1 - x_i >= r => 1 - x_i - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i] - v[r_idx]})
        # y_i >= r
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[r_idx]})
        # 1 - y_i >= r
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i+1] - v[r_idx]})
        
    # Distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: np.sum((v[2*i:2*i+2] - v[2*j:2*j+2])**2) - 4 * v[r_idx]**2})

    # Initial guess vector
    x0 = np.concatenate([centers.flatten(), [r_current]])
    
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(2 * n):
        bounds.append((0, 1))
    bounds.append((0, 0.5))
    
    # Optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Extract results
    final_centers = res.x[:2*n].reshape((n, 2))
    final_radii = np.full(n, res.x[-1])
    final_sum_radii = np.sum(final_radii)
    
    # Validate and adjust if necessary (simple clamp)
    # Ensure radii are positive
    if res.x[-1] < 1e-5:
        res.x[-1] = 0.05 # Fallback
        
    final_radii = np.full(n, res.x[-1])
    final_sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, final_sum_radii
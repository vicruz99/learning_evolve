# sol_000196 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 320c78c6) state=896e6325 sum of radii=2.338825 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_loss(x):
    """
    Computes the objective function (negative sum of radii) plus penalty terms
    for constraint violations.
    
    Args:
        x: np.array of shape (3*N,) containing [x1, y1, r1, x2, y2, r2, ...]
        
    Returns:
        loss: float, value to minimize
    """
    # Extract coordinates and radii
    x_coords = x[0::3]
    y_coords = x[1::3]
    radii = x[2::3]
    
    # Objective: maximize sum of radii => minimize -sum(radii)
    loss = -np.sum(radii)
    
    # Penalty weight
    P = 1000.0
    
    # Boundary violations
    # 1. Left boundary: x - r >= 0 => r - x <= 0. Violation if r > x.
    viol1 = radii - x_coords
    mask1 = viol1 > 0
    if np.any(mask1):
        loss += P * np.sum((viol1[mask1])**2)
        
    # 2. Right boundary: x + r <= 1 => x + r - 1 <= 0. Violation if x + r > 1.
    viol2 = x_coords + radii - 1.0
    mask2 = viol2 > 0
    if np.any(mask2):
        loss += P * np.sum((viol2[mask2])**2)
        
    # 3. Bottom boundary: y - r >= 0 => r - y <= 0. Violation if r > y.
    viol3 = radii - y_coords
    mask3 = viol3 > 0
    if np.any(mask3):
        loss += P * np.sum((viol3[mask3])**2)
        
    # 4. Top boundary: y + r <= 1 => y + r - 1 <= 0. Violation if y + r > 1.
    viol4 = y_coords + radii - 1.0
    mask4 = viol4 > 0
    if np.any(mask4):
        loss += P * np.sum((viol4[mask4])**2)
        
    # Overlap violations
    # Constraint: dist >= r_i + r_j
    # Violation: max(0, r_i + r_j - dist)
    
    for i in range(N):
        ri = radii[i]
        xi = x_coords[i]
        yi = y_coords[i]
        
        for j in range(i + 1, N):
            rj = radii[j]
            xj = x_coords[j]
            yj = y_coords[j]
            
            min_dist = ri + rj
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                diff = min_dist - dist
                loss += P * (diff * diff)
                
    return loss

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    # Initialization
    # Generate centers in a hexagonal lattice pattern to start with a dense configuration
    r_spacing = 0.09
    points = []
    
    row = 0
    while len(points) < N:
        for col in range(10):
            x = r_spacing + col * 2 * r_spacing + (row % 2) * r_spacing
            y = r_spacing + row * (np.sqrt(3) * r_spacing)
            
            # Check if inside [0,1] roughly
            if x <= 1.0 and y <= 1.0:
                points.append((x, y))
            if len(points) >= N:
                break
        row += 1
        if row > 20:
            break
            
    # Ensure we have exactly N points
    while len(points) < N:
        points.append((0.5, 0.5))
        
    points = points[:N]
    
    # Initial radii: small enough to ensure no initial overlaps
    r_init = 0.02
    
    x0 = []
    for (px, py) in points:
        x0.extend([px, py, r_init])
    x0 = np.array(x0, dtype=float)
    
    # Bounds for optimization
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    
    # Optimization parameters
    options = {'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
    
    try:
        # Use L-BFGS-B for bound-constrained optimization
        res = minimize(compute_loss, x0, method='L-BFGS-B', bounds=bounds, options=options)
        best_x = res.x
    except Exception:
        best_x = x0
        
    # Extract centers and radii
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(N):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    # Post-processing to ensure strict validity
    
    # 1. Enforce boundary constraints by clamping radii
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        
        # Clamp radius to fit within square boundaries
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r + 1e-9:
            r = max_r
            radii[i] = r
            
    # 2. Resolve overlaps by scaling down radii
    # Iterative approach to handle multiple overlaps
    for iteration in range(500):
        overlap_found = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                # Check for overlap with tolerance
                if dist < sum_r - 1e-9:
                    # Scale radii down so that they touch (r_i + r_j = dist)
                    if sum_r > 1e-12:
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = 0.0
                        radii[j] = 0.0
                    overlap_found = True
        if not overlap_found:
            break
            
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

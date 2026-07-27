import numpy as np
import scipy.optimize as opt

def generate_hexagonal_centers(n):
    """Generates a hexagonal grid of points to initialize the packing."""
    # Estimate radius for density calculation
    r_est = 0.1
    points = []
    x, y = 0, 0
    while len(points) < n:
        # Check horizontal row capacity
        row_points = []
        while x <= 1 and len(points) + len(row_points) < n:
            if 0 <= x <= 1 and 0 <= y <= 1:
                row_points.append([x, y])
            x += r_est * 2
        if len(row_points) < n - len(points):
            points.extend(row_points)
        else:
            points.extend(row_points[:n - len(points)])
        
        # Move to next hexagonal row
        if len(points) < n:
            y += r_est * np.sqrt(3)
            x = r_est # Offset x for hexagonal pattern
        else:
            x = 0
    return np.array(points[:n])

def get_lp_radius_bounds(centers):
    """Defines A_ub and b_ub for the LP problem: max sum(r) subject to r_i + r_j <= dist_ij."""
    n = centers.shape[0]
    A_ub = np.zeros((0, n))
    b_ub = np.zeros((0))
    
    # Distance constraints: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub = np.vstack([A_ub, row])
            b_ub = np.append(b_ub, dist)
            
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        for val in [centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]]:
            if val < 0: val = 0
            row = np.zeros(n)
            row[i] = 1.0
            A_ub = np.vstack([A_ub, row])
            b_ub = np.append(b_ub, val)
            
    return A_ub, b_ub

def run_packing() -> tuple:
    n = 26
    # Initialize with a hexagonal grid
    centers = generate_hexagonal_centers(n)
    radii = np.zeros(n)
    
    iterations = 2000
    for step in range(iterations):
        # 1. Solve LP for optimal radii given current centers
        A_ub, b_ub = get_lp_radius_bounds(centers)
        c_obj = -np.ones(n) # Maximize sum(r)
        bounds = [(0, None) for _ in range(n)]
        
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            break
        radii = res.x
        
        # 2. Apply repulsion forces to centers to relax constraints
        force = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < 1e-6: 
                    dist = 1e-6
                    delta = centers[j] - centers[i]
                else:
                    delta = centers[j] - centers[i] / dist
                
                # Push apart if touching or overlapping
                threshold = radii[i] + radii[j]
                if dist <= threshold * 1.005:
                    repulsion_strength = (threshold - dist + 0.001) * 0.5
                    force[i] -= delta * repulsion_strength
                    force[j] += delta * repulsion_strength
            
            # Boundary repulsion
            for dim in range(2):
                if centers[i, dim] < radii[i]:
                    force[i, dim] += (radii[i] - centers[i, dim]) * 2
                elif centers[i, dim] > 1 - radii[i]:
                    force[i, dim] -= (centers[i, dim] - (1 - radii[i])) * 2

        # Update centers
        centers += force
        # Project back to [0,1]
        centers = np.clip(centers, 0, 1)
        
    # Final radius optimization after center movement
    A_ub, b_ub = get_lp_radius_bounds(centers)
    res = opt.linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, method='highs')
    if res.success:
        radii = res.x

    total_radius = np.sum(radii)
    return centers, radii, float(total_radius)
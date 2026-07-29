# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0a5b5ea2) state=754d219b sum of radii=1.725083 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_hex_grid(n):
    """Generates a hexagonal grid of points inside the unit square."""
    points = []
    row = 0
    x = 0.05
    y = 0.05
    while len(points) < n:
        while x < 1.0 and len(points) < n:
            if x >= 0 and x <= 1 and y >= 0 and y <= 1:
                points.append((x, y))
            x += 0.15  # Horizontal spacing
        y += 0.13  # Vertical spacing
        x = 0.05 + (0.075 if row % 2 == 1 else 0.0)  # Offset for hexagonal pattern
        row += 1
    return np.array(points[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # 1. Initialize centers in a hexagonal grid
    centers = get_hex_grid(n_circles)
    centers = np.clip(centers, 0.0, 1.0)
    
    # Initialize radii
    radii = np.full(n_circles, 0.001)
    
    # Simulation parameters
    max_iter = 10000
    initial_radius_step = 5e-6
    force_factor = 0.5
    
    # 2. Physics-based simulation to expand radii
    for iteration in range(max_iter):
        # Increase radii
        radius_step = initial_radius_step * (1.0 + 0.00001 * iteration)
        radii += radius_step
        
        # Calculate forces to resolve overlaps
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Normalized direction vector
                    dir_vec = diff / dist
                    # Apply repulsive force proportional to overlap
                    push = overlap * force_factor
                    forces[i] += dir_vec * push
                    forces[j] -= dir_vec * push
                elif dist < 1e-9:
                    # Handle exact overlap to avoid division by zero
                    forces[i] += np.random.rand(2) * 0.01
                    forces[j] -= np.random.rand(2) * 0.01
        
        # Boundary repulsion
        for i in range(n_circles):
            r = radii[i]
            x, y = centers[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) * force_factor
            # Right wall
            elif x > 1.0 - r:
                forces[i, 0] -= (x - (1.0 - r)) * force_factor
            
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) * force_factor
            # Top wall
            elif y > 1.0 - r:
                forces[i, 1] -= (y - (1.0 - r)) * force_factor
        
        # Update centers
        centers += forces
        
        # Clamp centers to valid range to prevent extreme overshooting
        centers = np.clip(centers, 0.0, 1.0)

    # 3. Final optimization using Linear Programming for radii
    # Maximize sum(radii)
    # Subject to:
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # r_i + r_j <= distance(c_i, c_j)
    
    c = -np.ones(n_circles) # Minimize -sum(r)
    
    A_ub = []
    b_ub = []
    
    # Wall constraints
    for i in range(n_circles):
        x, y = centers[i]
        # r_i <= x_i
        row = np.zeros(n_circles)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(x)
        
        # r_i <= 1 - x_i
        A_ub.append(row)
        b_ub.append(1.0 - x)
        
        # r_i <= y_i
        A_ub.append(row)
        b_ub.append(y)
        
        # r_i <= 1 - y_i
        A_ub.append(row)
        b_ub.append(1.0 - y)
        
    # Pairwise distance constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n_circles)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for radii (non-negative)
    bounds = [(0, None)] * n_circles
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        final_radii = res.x
        sum_radii = np.sum(final_radii)
    else:
        # Fallback to simulation result if LP fails
        final_radii = radii
        sum_radii = np.sum(radii)
        
    return centers, final_radii, float(sum_radii)

# Run the packing and print results
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")

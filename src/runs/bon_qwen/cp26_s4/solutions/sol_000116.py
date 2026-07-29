# sol_000116 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e52471dd) state=a86dfcbc sum of radii=2.498071 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def get_max_radii(centers):
    """
    Solves the LP to find maximum radii for given centers.
    Maximize sum(r) subject to r_i <= wall_dist_i and r_i + r_j <= dist_ij.
    """
    n = centers.shape[0]
    if n == 0:
        return np.array([])
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    c = -np.ones(n)
    
    # Constraints A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = centers[i]
        # Wall distances: r_i <= min(x, 1-x, y, 1-y)
        d_wall = min(x, 1.0 - x, y, 1.0 - y)
        
        # Constraint: 1*r_i <= d_wall
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(d_wall)
        
        # Pairwise constraints with j > i: r_i + r_j <= dist
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    try:
        # Try 'highs' method first (fast), fallback to 'simplex'
        try:
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except ValueError:
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')
        
        if res.success:
            return res.x
        else:
            return np.zeros(n)
    except Exception:
        return np.zeros(n)

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initialize centers
    # Create a grid of points to ensure good initial distribution
    xs = np.linspace(0.08, 0.92, 6)
    ys = np.linspace(0.08, 0.92, 5)
    
    grid_points = []
    for y in ys:
        for x in xs:
            grid_points.append([x, y])
    
    # Shuffle and pick 26 points to break symmetry and allow optimization to find better layout
    indices = np.random.permutation(len(grid_points))[:n]
    centers = np.array(grid_points)[indices]
    
    # Optimization parameters
    iterations = 1000
    step_size = 0.01
    tol = 1e-5
    
    for step in range(iterations):
        radii = get_max_radii(centers)
        
        forces = np.zeros((n, 2))
        
        for i in range(n):
            x, y = centers[i]
            r_i = radii[i]
            
            # Wall forces: Push away from walls if constrained
            if x - r_i < tol:
                forces[i, 0] += 1.0
            if (1.0 - x) - r_i < tol:
                forces[i, 0] -= 1.0
            if y - r_i < tol:
                forces[i, 1] += 1.0
            if (1.0 - y) - r_i < tol:
                forces[i, 1] -= 1.0
                
            # Neighbor forces: Repel if touching (constraint is tight)
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.hypot(dx, dy)
                if dist < 1e-9:
                    dist = 1e-9
                
                r_sum = radii[i] + radii[j]
                
                # Check if constraint is active (tight)
                # Constraint: r_i + r_j <= dist. Active if dist - (r_i + r_j) is small.
                slack = dist - r_sum
                if slack < tol:
                    dir_x = dx / dist
                    dir_y = dy / dist
                    # Repulsion force to increase distance
                    forces[i, 0] += dir_x
                    forces[i, 1] += dir_y
                    forces[j, 0] -= dir_x
                    forces[j, 1] -= dir_y
        
        # Update centers
        # Decay step size for convergence
        current_step = step_size * (0.998 ** (step / 10.0))
        centers += current_step * forces
        
        # Clip centers to [0, 1] to maintain validity
        centers = np.clip(centers, 0.0, 1.0)
        
        # Occasional jitter to escape local minima
        if step % 100 == 0 and step < 500:
            centers += np.random.randn(n, 2) * 0.001
            centers = np.clip(centers, 0.0, 1.0)

    # Final radii computation for the optimized centers
    radii = get_max_radii(centers)
    total_sum = np.sum(radii)
    
    return centers, radii, total_sum

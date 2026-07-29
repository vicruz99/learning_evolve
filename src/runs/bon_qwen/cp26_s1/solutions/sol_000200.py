# sol_000200 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 263f0241) state=8ee48c48 sum of radii=2.575730 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_initial_centers(n):
    """
    Generates initial centers in a hexagonal-like pattern to start close to optimal.
    """
    ys = np.linspace(0.1, 0.9, 5)
    centers = []
    
    # Row 0: 5 circles
    xs = np.linspace(0.1, 0.9, 5)
    centers.extend(zip(xs, [ys[0]]*5))
    
    # Row 1: 6 circles (shifted)
    xs = np.linspace(0.05, 0.95, 6)
    centers.extend(zip(xs, [ys[1]]*6))
    
    # Row 2: 5 circles
    xs = np.linspace(0.1, 0.9, 5)
    centers.extend(zip(xs, [ys[2]]*5))
    
    # Row 3: 6 circles (shifted)
    xs = np.linspace(0.05, 0.95, 6)
    centers.extend(zip(xs, [ys[3]]*6))
    
    # Row 4: 4 circles
    xs = np.linspace(0.2, 0.8, 4)
    centers.extend(zip(xs, [ys[4]]*4))
    
    return np.array(centers)

def solve_radii(centers):
    """
    Solves the Linear Program to find max radii for fixed centers.
    Maximize sum(r) s.t. r_i + r_j <= dist(i,j) and r_i <= boundary_dist.
    """
    n = len(centers)
    c_obj = np.ones(n) * -1.0  # Minimizing negative sum is maximizing sum
    
    A_ub = []
    b_ub = []
    
    # 1. Boundary Constraints: r_i <= distance to edges
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n)
        row[i] = 1.0
        
        # x - r >= 0 => r <= x
        A_ub.append(row); b_ub.append(x)
        # r + x <= 1 => r <= 1 - x
        A_ub.append(row); b_ub.append(1 - x)
        # y - r >= 0 => r <= y
        A_ub.append(row); b_ub.append(y)
        # r + y <= 1 => r <= 1 - y
        A_ub.append(row); b_ub.append(1 - y)
        
    # 2. Pairwise Constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    # Use HiGHS solver for efficiency
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback to small radii if LP fails
        return np.full(n, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing to maximize sum of radii.
    """
    n = 26
    centers = get_initial_centers(n)
    
    alpha = 0.03      # Initial step size
    tol_active = 1e-4 # Threshold to consider a constraint 'active'
    max_iter = 2000   # Max iterations
    
    best_sum = -1.0
    best_centers = centers.copy()
    best_radii = None
    
    for step in range(max_iter):
        radii = solve_radii(centers)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Calculate forces based on active constraints
        forces = np.zeros_like(centers)
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # 1. Boundary Repulsion
            # If radius is constrained by left wall (x=0), push right
            if x - r < tol_active: forces[i, 0] += 1.0
            # If radius is constrained by right wall (x=1), push left
            if (1 - x) - r < tol_active: forces[i, 0] -= 1.0
            # If radius is constrained by bottom wall (y=0), push up
            if y - r < tol_active: forces[i, 1] += 1.0
            # If radius is constrained by top wall (y=1), push down
            if (1 - y) - r < tol_active: forces[i, 1] -= 1.0
            
            # 2. Neighbor Repulsion
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # If the sum of radii is close to the distance, the constraint is active
                if dist - (radii[i] + radii[j]) < tol_active:
                    if dist > 1e-9:
                        # Direction from j to i
                        direction = (centers[i] - centers[j]) / dist
                        forces[i] += direction
                        forces[j] -= direction
        
        # Update centers
        centers += alpha * forces
        
        # Clamp to unit square [0, 1]
        centers = np.clip(centers, 0.0, 1.0)
        
        # Decay step size slowly
        alpha *= 0.999
        
    return best_centers, best_radii, float(np.sum(best_radii))

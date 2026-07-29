# sol_000166 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 724447fa) state=8892c4a1 sum of radii=0.434330 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    np.random.seed(42)
    N = 26
    
    # 1. Initialize centers in a hexagonal lattice pattern
    centers = []
    count = 0
    y = 0
    while count < N:
        x = 0
        shift = (y * np.sqrt(3) / 2) % 1  # Offset rows for hex packing
        if int(y * np.sqrt(3) / 2) % 2 == 1:
            shift = 0.5 * (1.0 / (N // 5 + 1)) # Simple approximation of shift
        
        # Generate points in a row
        while x < 1.0:
            if count < N:
                centers.append([x, y])
                count += 1
            x += 2.0 * 0.1 # Diameter of approx circle
            
        y += 0.1 * np.sqrt(3) # Vertical spacing
        
    centers = np.array(centers)
    
    # Randomize slightly to break symmetry and avoid local minima
    centers += np.random.uniform(-0.05, 0.05, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)

    # 2. Force-Directed Layout to maximize separation
    for step in range(2000):
        forces = np.zeros_like(centers)
        repulsion = 0.5
        softening = 0.01
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 0.01: dist = 0.01
                f = repulsion / (dist**2)
                forces[i] += f * diff / dist
                forces[j] -= f * diff / dist

        # Boundary forces
        for i in range(N):
            for dim in range(2):
                if centers[i][dim] < 0.05:
                    forces[i][dim] += 5.0 * (0.05 - centers[i][dim])
                elif centers[i][dim] > 0.95:
                    forces[i][dim] -= 5.0 * (centers[i][dim] - 0.95)

        centers += forces * 0.01
        centers = np.clip(centers, 0.001, 0.999)

    # 3. Optimize Radii using Linear Programming
    # Maximize sum(r_i)
    # s.t. r_i + r_j <= distance(i, j)
    #      r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    c = -np.ones(N) # Maximize sum(r) -> Minimize -sum(r)
    A_ub = []
    b_ub = []
    
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = [0] * N
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    # Boundary constraints
    for i in range(N):
        bound = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        row = [0] * N
        row[i] = 1
        A_ub.append(row)
        b_ub.append(bound)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for radii: [0, infinity)
    bounds = [(0, None) for _ in range(N)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
    else:
        # Fallback if LP fails: simple geometric assignment
        radii = np.zeros(N)
        for i in range(N):
            b = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            min_d = float('inf')
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_d: min_d = d
            radii[i] = min(b, min_d / 2)
            
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

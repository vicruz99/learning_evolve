# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9227c4d6) state=ebd1e4e0 sum of radii=2.459019 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initialize Centers (Hexagonal-like layout)
    n = 26
    centers = np.zeros((n, 2))
    
    # Pattern: 5, 4, 5, 4, 5, 3 circles per row
    row_counts = [5, 4, 5, 4, 5, 3]
    d_guess = 0.21  # Initial guess for diameter
    h_guess = d_guess * np.sqrt(3) / 2
    
    idx = 0
    for r_idx, count in enumerate(row_counts):
        y = r_idx * h_guess + d_guess / 2
        if r_idx % 2 == 1:
            # Offset rows
            x_start = d_guess / 2 + d_guess / 2
        else:
            x_start = d_guess / 2
            
        for c_idx in range(count):
            if idx < n:
                centers[idx, 0] = x_start + c_idx * d_guess
                centers[idx, 1] = y
                idx += 1

    # 2. Initial Radii
    radii = np.full(n, 0.1)

    # 3. Optimization Loop
    for iteration in range(10000):
        # Check and resolve overlaps and boundary violations
        improved = False
        
        # Boundary constraint check
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            
            # Push inside boundaries
            if x - r < 0: centers[i, 0] = r
            if x + r > 1: centers[i, 0] = 1 - r
            if y - r < 0: centers[i, 1] = r
            if y + r > 1: centers[i, 1] = 1 - r
            
        # Inter-circle constraint check and repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                req_dist = radii[i] + radii[j]
                
                if dist < req_dist and dist > 1e-9:
                    # Overlap detected, push apart
                    overlap = req_dist - dist
                    # Push back proportional to radii
                    shift = overlap / 2
                    direction = (centers[i] - centers[j]) / dist
                    
                    centers[i] += direction * shift
                    centers[j] -= direction * shift
                    improved = True

        # Boundary re-check after repulsion
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            if x - r < 0: centers[i, 0] = r
            if x + r > 1: centers[i, 0] = 1 - r
            if y - r < 0: centers[i, 1] = r
            if y + r > 1: centers[i, 1] = 1 - r

        # Expansion step: try to increase radii if space allows
        if not improved:
            # Simple local expansion: move away from neighbors slightly to create space
            for i in range(n):
                move_vec = np.array([0.0, 0.0])
                for j in range(n):
                    if i == j: continue
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    if dist < radii[i] + radii[j] + 0.005: # If close
                        move_vec -= (centers[i] - centers[j]) / (dist**2 + 1e-9)
                
                if np.linalg.norm(move_vec) > 1e-6:
                    centers[i] += 0.001 * move_vec / np.linalg.norm(move_vec)
            
            # Re-enforce boundaries
            for i in range(n):
                r = radii[i]
                if centers[i, 0] < r: centers[i, 0] = r
                if centers[i, 0] > 1 - r: centers[i, 0] = 1 - r
                if centers[i, 1] < r: centers[i, 1] = r
                if centers[i, 1] > 1 - r: centers[i, 1] = 1 - r
            
            # Increase radii slightly
            radii += 0.0005

    # 4. Local Optimization (LP) to maximize radii for final centers
    m = n
    A_ub = []
    b_ub = []
    
    # Constraints: r_i + r_j <= dist(i, j)
    # We convert to -r_i - r_j >= -dist(i, j) => r_i + r_j <= dist
    # Standard form for linprog (min c^T x): -sum(r) => max sum(r)
    # Constraints A_ub @ x <= b_ub
    
    for i in range(m):
        for j in range(i + 1, m):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = [0] * m
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(m):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        
        row = [0] * m
        row[i] = 1
        A_ub.append(row)
        b_ub.append(max_r)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Maximize sum(r) -> Minimize -sum(r)
    c = np.ones(m) * -1.0
    bounds = [(0, None) for _ in range(m)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
        else:
            # Fallback if LP fails (rare)
            pass
    except Exception:
        pass

    # Final sanity clamp for numerical stability
    radii = np.clip(radii, 0, 0.5)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r: centers[i, 0] = r
        if x > 1 - r: centers[i, 0] = 1 - r
        if y < r: centers[i, 1] = r
        if y > 1 - r: centers[i, 1] = 1 - r

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# sol_000191 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bafdbd7e) state=5c2c5c55 sum of radii=1.524807 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing():
    n = 26
    # Initialize centers on a hexagonal grid
    centers = []
    r_init = 0.08
    row = 0
    col = 0
    while len(centers) < n:
        x = col * 2 * r_init + (row % 2) * r_init
        y = row * np.sqrt(3) * r_init
        if x <= 1 - r_init and y <= 1 - r_init:
            centers.append([x, y])
        col += 1
        if x > 1 - r_init:
            row += 1
            col = 0
    centers = np.array(centers[:n])
    
    # Iterative optimization: LP for radii, push centers apart
    alpha = 0.03
    for _ in range(300):
        c_obj = -np.ones(n)
        A_ub = []
        b_ub = []
        
        for i in range(n):
            for j in range(i+1, n):
                row_vec = np.zeros(n)
                row_vec[i] = 1.0
                row_vec[j] = 1.0
                A_ub.append(row_vec)
                b_ub.append(np.sqrt(np.sum((centers[i] - centers[j])**2)))
                
        for i in range(n):
            row_vec = np.zeros(n)
            row_vec[i] = 1.0
            A_ub.append(row_vec); b_ub.append(centers[i, 0])
            A_ub.append(row_vec); b_ub.append(1.0 - centers[i, 0])
            A_ub.append(row_vec); b_ub.append(centers[i, 1])
            A_ub.append(row_vec); b_ub.append(1.0 - centers[i, 1])
            
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        radii = res.x
        
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d > 1e-8 and radii[i] + radii[j] > d - 1e-7:
                    dir = (centers[i] - centers[j]) / d
                    centers[i] += dir * alpha
                    centers[j] -= dir * alpha
                    moved = True
                    
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
            
        if not moved:
            alpha *= 0.5
            if alpha < 1e-7:
                break
                
    # Final LP to ensure strict feasibility
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i+1, n):
            row_vec = np.zeros(n)
            row_vec[i] = 1.0
            row_vec[j] = 1.0
            A_ub.append(row_vec)
            b_ub.append(np.sqrt(np.sum((centers[i] - centers[j])**2)))
    for i in range(n):
        row_vec = np.zeros(n)
        row_vec[i] = 1.0
        A_ub.append(row_vec); b_ub.append(centers[i, 0])
        A_ub.append(row_vec); b_ub.append(1.0 - centers[i, 0])
        A_ub.append(row_vec); b_ub.append(centers[i, 1])
        A_ub.append(row_vec); b_ub.append(1.0 - centers[i, 1])
        
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    radii = np.maximum(res.x, 0)
    
    return centers, radii, np.sum(radii)

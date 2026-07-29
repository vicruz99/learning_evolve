# sol_000277 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c64acbd5) state=ca17602c sum of radii=2.502669 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def run_packing():
    n = 26
    
    # 1. Initialize centers
    # Generate a grid of points
    candidates = []
    step = 0.05
    for x in np.arange(0.05, 1.05, step):
        for y in np.arange(0.05, 1.05, step):
            candidates.append([x, y])
    
    # Select 26 points with max dispersion
    # Simple greedy
    selected = [candidates[0]]
    remaining = candidates[1:]
    
    while len(selected) < n and remaining:
        best_p = None
        best_min_dist = -1.0
        for p in remaining:
            min_d = float('inf')
            for s in selected:
                d = math.hypot(p[0]-s[0], p[1]-s[1])
                if d < min_d:
                    min_d = d
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_p = p
        
        if best_p:
            selected.append(best_p)
            # Remove best_p from remaining
            # List remove is O(N), but N is small
            remaining.remove(best_p)
        else:
            break
            
    centers = np.array(selected)
    
    # 2. Optimization loop
    # We will try to improve centers to maximize sum of radii
    # We use a heuristic: move centers apart if they are "tight"
    
    # Initial radii calculation
    # Solve LP to get initial radii
    def solve_radii(centers):
        n_c = centers.shape[0]
        c_obj = np.ones(n_c) * -1.0
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints
        for i in range(n_c):
            x, y = centers[i]
            bound = min(x, 1-x, y, 1-y)
            row = np.zeros(n_c)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bound)
            
        # Pairwise constraints
        for i in range(n_c):
            for j in range(i+1, n_c):
                dist = math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1])
                row = np.zeros(n_c)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None) for _ in range(n_c)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                return np.zeros(n_c)
        except:
            return np.zeros(n_c)

    radii = solve_radii(centers)
    
    # Optimization parameters
    max_iter = 100
    alpha = 0.02 # Step size
    
    for step in range(max_iter):
        # Recompute radii
        radii = solve_radii(centers)
        sum_r = np.sum(radii)
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Check constraints activity
        # We need to know which constraints are tight.
        # Since we don't have duals easily, we check slack.
        
        # Boundary slacks
        bound_slacks = []
        for i in range(n):
            x, y = centers[i]
            b = min(x, 1-x, y, 1-y)
            bound_slacks.append(b - radii[i])
            
        # Pairwise slacks
        # We only need to check pairs that are close?
        # Or all. 26*25/2 = 325 checks. Fast.
        
        for i in range(n):
            # Boundary force
            if bound_slacks[i] < 1e-6:
                # Tight boundary
                x, y = centers[i]
                # Push away from nearest wall
                dists = [x, 1-x, y, 1-y]
                min_idx = np.argmin(dists)
                if min_idx == 0: fx, fy = 1.0, 0.0 # x is small, push right
                elif min_idx == 1: fx, fy = -1.0, 0.0
                elif min_idx == 2: fx, fy = 0.0, 1.0
                else: fx, fy = 0.0, -1.0
                forces[i] += np.array([fx, fy]) * 0.5
            
            for j in range(i+1, n):
                d = math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1])
                slack = d - (radii[i] + radii[j])
                if slack < 1e-6:
                    # Tight pair
                    if d > 1e-9:
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        norm = math.hypot(dx, dy)
                        fx, fy = dx/norm, dy/norm
                        # Force magnitude proportional to tightness?
                        # Just constant
                        forces[i] += np.array([fx, fy]) * 0.5
                        forces[j] -= np.array([fx, fy]) * 0.5
                    else:
                        # Coincident
                        fx, fy = np.random.uniform(-1, 1, 2)
                        norm = math.hypot(fx, fy)
                        if norm > 0: fx, fy = fx/norm, fy/norm
                        forces[i] += np.array([fx, fy]) * 0.5
                        forces[j] -= np.array([fx, fy]) * 0.5

        # Update centers
        centers += forces * alpha
        
        # Clip
        centers = np.clip(centers, 1e-5, 1 - 1e-5)
        
        # Reduce alpha
        if step % 20 == 0:
            alpha *= 0.9

    # Final radii
    radii = solve_radii(centers)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

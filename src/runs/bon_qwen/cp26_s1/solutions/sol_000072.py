# sol_000072 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfef56bb) state=544f15f4 sum of radii=1.580000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def compute_forces(centers, radii):
    n = len(centers)
    forces = np.zeros_like(centers)
    
    for i in range(n):
        # Wall repulsion
        for dim in range(2):
            if centers[i, dim] < radii[i]:
                forces[i, dim] += (radii[i] - centers[i, dim]) ** 2
            elif centers[i, dim] > 1.0 - radii[i]:
                forces[i, dim] -= (centers[i, dim] - (1.0 - radii[i])) ** 2

        # Pairwise repulsion
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                overlap = min_dist - dist
                if dist > 1e-8:
                    force_vec = (overlap ** 2) * (diff / dist)
                else:
                    # Avoid division by zero with random direction
                    force_vec = overlap ** 2 * np.random.rand(2)
                forces[i] += force_vec
                forces[j] -= force_vec
                
    return forces

def solve_radii_lp(centers):
    n = len(centers)
    # Objective: maximize sum(radii) => minimize -sum(radii)
    c_obj = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        for d in range(2):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(centers[i, d])
            
            A_ub.append(row)
            b_ub.append(1.0 - centers[i, d])
            
    # Pairwise constraints: r_i + r_j <= distance(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None) for _ in range(n)]
    
    try:
        # Use HiGHS solver if available, otherwise fallback
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    # Fallback: safe small radii
    r = np.full(n, 0.01)
    return r, np.sum(r)

def objective_function(x, n, radii):
    c = x.reshape(n, 2)
    energy = 0.0
    
    for i in range(n):
        for d in range(2):
            if c[i, d] < radii[i]:
                energy += (radii[i] - c[i, d]) ** 2
            if c[i, d] > 1.0 - radii[i]:
                energy += (c[i, d] - (1.0 - radii[i])) ** 2
                
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(c[i] - c[j])
            if d < radii[i] + radii[j]:
                energy += (radii[i] + radii[j] - d) ** 2
                
    return energy

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers in a hexagonal-like grid
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        for col in range(6):
            if idx == n:
                break
            x = 0.12 + col * 0.14
            y = 0.12 + row * 0.18 + (0.07 if row % 2 == 1 else 0.0)
            centers[idx] = [x, y]
            idx += 1
            
    # Initial radii estimate
    radii, _ = solve_radii_lp(centers)
    
    # 2. Force-directed simulation to spread centers
    dt = 0.4
    for step in range(2500):
        forces = compute_forces(centers, radii)
        centers += forces * dt
        # Keep centers strictly inside to avoid numerical issues in LP
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
        
        # Periodically update radii via LP to guide forces
        if step % 50 == 0:
            radii, _ = solve_radii_lp(centers)
            
    # 3. Alternating local optimization
    # Fixes centers -> optimizes radii (LP) -> fixes radii -> optimizes centers (Gradient)
    for _ in range(15):
        radii, _ = solve_radii_lp(centers)
        
        bounds_opt = [(0.0, 1.0) for _ in range(2 * n)]
        res = minimize(
            objective_function, 
            centers.flatten(), 
            args=(n, radii),
            method='L-BFGS-B',
            bounds=bounds_opt,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        centers = res.x.reshape(n, 2)
        
    # Final precise radius calculation
    radii, total_sum = solve_radii_lp(centers)
    
    return centers, radii, total_sum

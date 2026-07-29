# sol_000136 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc363b95) state=2d0af2de sum of radii=2.134420 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_optimal_radii(centers):
    """Solve LP to maximize sum of radii for fixed center positions."""
    n = centers.shape[0]
    c = -np.ones(n)  # Minimize -sum(r) <=> Maximize sum(r)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n)
        row[i] = 1.0
        # r_i <= x_i
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
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            # r_i + r_j <= dist_ij
            A_ub.append(row)
            b_ub.append(dist)
            
    bounds = [(0, None) for _ in range(n)]
    try:
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers on a hexagonal lattice
    pts = []
    r_init = 0.1
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    row = 0
    while len(pts) < n:
        for col in range(7):
            x = col * dx + (row % 2) * r_init + 0.05
            y = row * dy + 0.05
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
        row += 1
    centers = np.array(pts[:n])
    
    best_centers = centers.copy()
    best_radii = np.ones(n) * 0.01
    best_sum = 0.0
    
    # 2. Iterative optimization
    for step in range(400):
        radii, current_sum = get_optimal_radii(centers)
        if radii is not None and current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Compute repulsion forces
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                target = radii[i] + radii[j]
                if dist < target and dist > 1e-6:
                    overlap = target - dist
                    f = overlap * 30.0
                    fx = (dx / dist) * f
                    fy = (dy / dist) * f
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    
            x, y = centers[i]
            r = radii[i]
            # Boundary repulsion
            if x - r < 0: forces[i, 0] += (x - r) * 40.0
            elif x + r > 1: forces[i, 0] -= (x + r - 1) * 40.0
            if y - r < 0: forces[i, 1] += (y - r) * 40.0
            elif y + r > 1: forces[i, 1] -= (y + r - 1) * 40.0
            
        # Update positions with cooling step size
        step_size = 0.002 * (1.0 - step / 600.0)
        centers += forces * step_size
        centers = np.clip(centers, 0.0005, 0.9995)
        
        # Occasional perturbation to escape local minima
        if step % 50 == 0 and step > 0:
            centers += np.random.randn(n, 2) * 0.001
            centers = np.clip(centers, 0.0005, 0.9995)
            
    # Ensure final radii match final centers exactly
    final_radii, _ = get_optimal_radii(best_centers)
    if final_radii is not None:
        best_radii = final_radii
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)

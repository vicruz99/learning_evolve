# sol_000076 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05693c56) state=649a980e sum of radii=1.165853 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_radii(centers):
    n = centers.shape[0]
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n)
        row[i] = 1.0
        # r_i <= x_i
        A_ub.append(row); b_ub.append(x)
        # r_i <= 1 - x_i
        A_ub.append(row); b_ub.append(1.0 - x)
        # r_i <= y_i
        A_ub.append(row); b_ub.append(y)
        # r_i <= 1 - y_i
        A_ub.append(row); b_ub.append(1.0 - y)
        
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d < 1e-9:
                d = 1e-9
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    try:
        res = linprog(c, A_ub=np.array(A_ub, dtype=float), b_ub=np.array(b_ub, dtype=float), 
                      bounds=[(0, None)] * n, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 0.01)

def compute_forces(centers, radii, stiffness):
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    diff = centers[np.newaxis, :] - centers[:, np.newaxis]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    rad_sum = radii[:, None] + radii[None, :]
    overlap = np.maximum(rad_sum - dist, 0.0)
    
    safe_dist = np.where(dist < 1e-9, 1.0, dist)
    u_vec = diff / safe_dist[:, :, np.newaxis]
    
    force_mat = -stiffness * overlap[:, :, np.newaxis] * u_vec
    forces += np.sum(force_mat, axis=0)
    
    for k in range(2):
        low = np.maximum(0.0, radii - centers[:, k])
        high = np.maximum(0.0, centers[:, k] - (1.0 - radii))
        forces[:, k] += stiffness * (low - high)
        
    return forces

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    centers = rng.random((n, 2)) * 0.8 + 0.1
    
    radii = solve_radii(centers)
    
    vel = np.zeros_like(centers)
    stiffness = 200.0
    damping = 0.93
    dt = 0.005
    
    for it in range(8000):
        if it % 3 == 0:
            radii = solve_radii(centers)
            
        forces = compute_forces(centers, radii, stiffness)
        vel = damping * vel + forces * dt
        centers += vel
        centers = np.clip(centers, 0.0, 1.0)
        
        if it % 1000 == 0 and it > 0:
            stiffness *= 0.85
            dt *= 0.9
            
    radii = solve_radii(centers)
    return centers, radii, float(np.sum(radii))

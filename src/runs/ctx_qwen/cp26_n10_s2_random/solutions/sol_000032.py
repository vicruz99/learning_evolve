# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state 58c90071) state=75045fb5 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    np.random.seed(42)
    
    # 1. Initial Configuration: Hexagonal Lattice
    # Hexagonal arrangement naturally packs circles more densely than square grids
    centers = np.zeros((N, 2))
    idx = 0
    r_est = 0.09
    dy = r_est * np.sqrt(3)
    dx = 2 * r_est
    
    row = 0
    y = r_est
    while idx < N and y + r_est <= 1.0:
        # Shift odd rows to nest into the gaps of even rows
        x_start = r_est + (dx / 2.0 if row % 2 == 1 else 0.0)
        x = x_start
        while x + r_est <= 1.0 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += dx
        y += dy
        row += 1
        
    # Fill remaining circles if the hex grid didn't produce exactly N
    while idx < N:
        cx, cy = np.random.uniform(r_est, 1-r_est, 2)
        centers[idx] = [cx, cy]
        idx += 1
        
    radii = np.full(N, r_est)
    
    # 2. Force-directed expansion simulation
    # Grows radii iteratively while using repulsion forces to resolve overlaps
    velocities = np.zeros_like(centers)
    repulsion_k = 5000.0
    boundary_k = 10000.0
    step_size = 0.005
    damping = 0.7
    
    for step in range(20000):
        # Calculate minimum gap to safely grow radii
        gaps = np.ones(N) * 1.0
        gaps = np.minimum(gaps, centers[:, 0] - radii)
        gaps = np.minimum(gaps, 1.0 - centers[:, 0] - radii)
        gaps = np.minimum(gaps, centers[:, 1] - radii)
        gaps = np.minimum(gaps, 1.0 - centers[:, 1] - radii)
        
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        gaps_mat = dists - (radii[:, None] + radii[None, :])
        gaps = np.minimum(gaps, np.min(gaps_mat, axis=1))
        min_gap = np.min(gaps)
        
        # Grow radii proportionally to available space
        growth = min(2e-5, min_gap * 0.3)
        if growth > 0:
            radii += growth
            
        # Compute repulsion forces (vectorized)
        overlap = np.maximum(0, (radii[:, None] + radii[None, :]) - dists)
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        dirs = diffs / safe_dists[:, :, np.newaxis]
        force_mag = (overlap * repulsion_k / safe_dists)
        forces = np.sum(force_mag[:, :, np.newaxis] * dirs, axis=1)
        
        # Apply boundary repulsion forces
        for i in range(N):
            if centers[i,0] < radii[i]:
                forces[i,0] += boundary_k * (radii[i] - centers[i,0])
            elif centers[i,0] > 1.0 - radii[i]:
                forces[i,0] -= boundary_k * (centers[i,0] - (1.0 - radii[i]))
            if centers[i,1] < radii[i]:
                forces[i,1] += boundary_k * (radii[i] - centers[i,1])
            elif centers[i,1] > 1.0 - radii[i]:
                forces[i,1] -= boundary_k * (centers[i,1] - (1.0 - radii[i]))
                
        # Update dynamics
        velocities = damping * velocities + step_size * forces
        centers += velocities
        centers = np.clip(centers, 0.0, 1.0)
        
        # Adaptive step size decay
        if step % 1000 == 0 and step > 0:
            step_size *= 0.95

    # 3. SLSQP Local Optimization to polish positions and radii
    def obj(v):
        return -np.sum(v[2::3])
        
    def constr(v):
        c = np.zeros(N * 4 + N*(N-1)//2)
        idx_c = 0
        for i in range(N):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            c[idx_c] = x - r; idx_c+=1
            c[idx_c] = 1 - x - r; idx_c+=1
            c[idx_c] = y - r; idx_c+=1
            c[idx_c] = 1 - y - r; idx_c+=1
        idx2 = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dr = v[3*i+2] + v[3*j+2]
                c[idx_c + idx2] = dx**2 + dy**2 - dr**2
                idx2 += 1
        return c

    x0 = np.zeros(N*3)
    for i in range(N):
        x0[3*i] = centers[i,0]
        x0[3*i+1] = centers[i,1]
        x0[3*i+2] = radii[i]
        
    # Ensure initial guess strictly respects bounds
    x0[::3] = np.clip(x0[::3], 0, 1)
    x0[1::3] = np.clip(x0[1::3], 0, 1)
    x0[2::3] = np.clip(x0[2::3], 0, 0.5)
        
    bounds = [(0,1)]*(2*N) + [(0, 0.5)]*N
    cons = {'type': 'ineq', 'fun': constr}
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False})
        x_opt = res.x
    except Exception:
        x_opt = x0
        
    final_centers = x_opt.reshape(N, 3)[:, :2]
    final_radii = x_opt.reshape(N, 3)[:, 2]
    
    # Final safety shrink to guarantee validation passes within 1e-12 tolerance
    max_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(final_centers[i,0]-final_centers[j,0], final_centers[i,1]-final_centers[j,1])
            v = final_radii[i] + final_radii[j] - d
            if v > max_viol: max_viol = v
        x, y, r = final_centers[i,0], final_centers[i,1], final_radii[i]
        if x < r: max_viol = max(max_viol, r - x)
        if x > 1-r: max_viol = max(max_viol, x - (1-r))
        if y < r: max_viol = max(max_viol, r - y)
        if y > 1-r: max_viol = max(max_viol, y - (1-r))
        
    if max_viol > 0:
        final_radii -= (max_viol + 1e-7)
        final_radii = np.maximum(final_radii, 0.0)
        
    return final_centers, final_radii, float(np.sum(final_radii))

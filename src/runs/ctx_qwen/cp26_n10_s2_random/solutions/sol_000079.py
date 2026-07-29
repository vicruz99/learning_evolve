# sol_000079 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000058 (state f7fedeb3) state=a7c04b0a sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def simulate_packing(seed, steps=5000):
    """Force-directed simulation to find a dense local packing."""
    rng = np.random.default_rng(seed)
    N = N_CIRCLES
    
    cfg_type = seed % 3
    if cfg_type == 0:
        centers = rng.uniform(0.2, 0.8, (N, 2))
    elif cfg_type == 1:
        centers = np.zeros((N, 2))
        r_est = 0.09
        dy = r_est * np.sqrt(3)
        row = 0
        cnt = 0
        while cnt < N:
            y = r_est + row * dy
            if y + r_est > 1.0: break
            offset = r_est if row % 2 == 1 else 0.0
            x = r_est + offset
            while x + r_est <= 1.0 and cnt < N:
                centers[cnt] = [x, y]
                x += 2 * r_est
                cnt += 1
            row += 1
        while cnt < N:
            centers[cnt] = rng.uniform(0.2, 0.8, 2)
            cnt += 1
        centers += rng.normal(0, 0.005, (N, 2))
    else:
        gs = np.linspace(0.15, 0.85, 6)
        grid = np.array(np.meshgrid(gs, gs)).T.reshape(-1, 2)
        centers = grid[:N]
        centers += rng.normal(0, 0.005, (N, 2))
        
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(N, 0.05)
    velocities = np.zeros((N, 2))
    
    dt = 0.004
    damping = 0.96
    grow_factor = 1.00008
    
    for _ in range(steps):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0, sum_r - dists)
        
        dist_safe = np.where(dists < 1e-9, 1e-9, dists)
        dir_unit = diff / dist_safe[:, :, np.newaxis]
        
        force_mag = overlap * 150.0
        forces = np.sum(force_mag[:, :, np.newaxis] * dir_unit, axis=1)
        
        x, y = centers[:, 0], centers[:, 1]
        bnd_forces = np.zeros_like(centers)
        bnd_forces[:, 0] += np.maximum(0, radii - x) * 150.0
        bnd_forces[:, 0] -= np.maximum(0, x + radii - 1.0) * 150.0
        bnd_forces[:, 1] += np.maximum(0, radii - y) * 150.0
        bnd_forces[:, 1] -= np.maximum(0, y + radii - 1.0) * 150.0
        
        total_forces = forces + bnd_forces
        
        velocities = damping * velocities + total_forces * dt
        centers += velocities
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        
        max_ov = np.max(overlap)
        max_bnd = max(np.max(np.maximum(0, radii - x)), np.max(np.maximum(0, x + radii - 1.0)),
                      np.max(np.maximum(0, radii - y)), np.max(np.maximum(0, y + radii - 1.0)))
        
        if max_ov < 1e-6 and max_bnd < 1e-6:
            radii *= grow_factor
            
    return centers, radii

def slsqp_obj(v):
    return -np.sum(v[2*N_CIRCLES:])

def slsqp_cons(v):
    c = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r = v[2*N_CIRCLES:]
    out = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    d = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(d, np.inf)
    mask = np.triu_indices(N_CIRCLES, k=1)
    out.append(d[mask] - (r[mask[0]] + r[mask[1]]))
    return np.concatenate(out)

def slsqp_polish(centers, radii):
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    try:
        res = minimize(slsqp_obj, x0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 3000, 'ftol': 1e-12})
        if res.success or res.nit > 100:
            return res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2), res.x[2*N_CIRCLES:]
    except Exception:
        pass
    return centers, radii

def repair_solution(centers, radii):
    N = N_CIRCLES
    radii = radii.copy()
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    ov = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] -= ov
                    radii[j] -= ov
                    changed = True
        for i in range(N):
            mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mx + 1e-12:
                radii[i] = mx
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Phase 1: Diverse simulations to find good spatial arrangements
    for i in range(20):
        c, r = simulate_packing(seed=rng.integers(0, 10000), steps=5000)
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_centers = c.copy()
            best_radii = r.copy()
            
    # Phase 2: Local polishing with perturbation to escape shallow minima
    candidates = [(best_centers.copy(), best_radii.copy())]
    for _ in range(5):
        pc = best_centers + rng.normal(0, 0.002, (N_CIRCLES, 2))
        pc = np.clip(pc, 0.05, 0.95)
        pr = best_radii * 0.99
        candidates.append((pc, pr))
        
    for c, r in candidates:
        c_opt, r_opt = slsqp_polish(c, r)
        if np.sum(r_opt) > np.sum(best_radii):
            best_centers = c_opt
            best_radii = r_opt
            
    # Phase 3: Strict repair to guarantee validation compliance
    best_radii = repair_solution(best_centers, best_radii)
    total_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, total_sum

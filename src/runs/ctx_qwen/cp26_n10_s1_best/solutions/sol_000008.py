# sol_000008 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d25b46ef) state=31daac10 sum of radii=1.840800 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(pos, r, n):
    """Computes squared penalty for overlaps and boundary violations."""
    penalty = 0.0
    x, y = pos[:, 0], pos[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
        xi, yi = x[i], y[i]
        ri = r[i]
        if xi < ri: penalty += (ri - xi) ** 2
        if xi > 1 - ri: penalty += (xi - (1 - ri)) ** 2
        if yi < ri: penalty += (ri - yi) ** 2
        if yi > 1 - ri: penalty += (yi - (1 - ri)) ** 2
    return penalty

def _scipy_obj(params, n, r):
    """Wrapper for scipy optimizer."""
    return compute_penalty(params.reshape(-1, 2), r, n)

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Stage 1: Expanding Bubbles Simulation
    cols = 6
    x = np.array([0.15 + (i % cols) * 0.14 + (0.07 if (i // cols) % 2 == 1 else 0.0) for i in range(n)])
    y = np.array([0.15 + (i // cols) * 0.14 for i in range(n)])
    r = np.full(n, 0.01)
    
    vx, vy = np.zeros(n), np.zeros(n)
    lr = 0.002
    mom = 0.65
    grow = 0.0008
    
    best_r = r.copy()
    best_pos = np.column_stack((x, y))
    
    for step in range(20000):
        fx, fy, pen = np.zeros(n), np.zeros(n), 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = x[i] - x[j], y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-8
                ov = r[i] + r[j] - d
                if ov > 0:
                    f = ov / d
                    fx[i] -= f * dx
                    fy[i] -= f * dy
                    fx[j] += f * dx
                    fy[j] += f * dy
                    pen += ov * ov
            if x[i] < r[i]:
                ov = r[i] - x[i]
                fx[i] += ov
                pen += ov * ov
            if x[i] > 1 - r[i]:
                ov = x[i] - (1 - r[i])
                fx[i] -= ov
                pen += ov * ov
            if y[i] < r[i]:
                ov = r[i] - y[i]
                fy[i] += ov
                pen += ov * ov
            if y[i] > 1 - r[i]:
                ov = y[i] - (1 - r[i])
                fy[i] -= ov
                pen += ov * ov
                
        vx = mom * vx - lr * fx
        vy = mom * vy - lr * fy
        x += vx
        y += vy
        x = np.clip(x, 1e-5, 1 - 1e-5)
        y = np.clip(y, 1e-5, 1 - 1e-5)
        
        # Feedback control for radius growth
        if pen < 1e-6:
            r += grow
        elif pen > 0.001:
            r -= grow * 0.3
        lr = np.clip(lr * (0.95 if pen > 0.001 else 1.001), 1e-4, 0.005)
        
        if np.sum(r) > np.sum(best_r) and pen < 1e-7:
            best_r = r.copy()
            best_pos = np.column_stack((x, y))

    # Stage 2: Gradient-based cleanup to strictly satisfy constraints
    target_r = best_r
    for _ in range(15):
        res = minimize(_scipy_obj, best_pos.flatten(), args=(n, target_r),
                       method='L-BFGS-B', bounds=[(0.0, 1.0)] * (2 * n),
                       options={'ftol': 1e-13, 'gtol': 1e-13, 'maxiter': 2000})
        final_pos = res.x.reshape(-1, 2)
        if res.fun < 1e-10:
            break
        target_r *= 0.995
        best_pos = final_pos

    return final_pos, target_r, np.sum(target_r)

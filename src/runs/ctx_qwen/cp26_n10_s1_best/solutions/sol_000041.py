# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000031 (state b051e300) state=d23961ea sum of radii=2.616291 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[2*N:])

def constraints(vars):
    """Compute boundary and non-overlap constraints as a single vector >= 0."""
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    b1 = centers[:, 0] - radii
    b2 = 1.0 - centers[:, 0] - radii
    b3 = centers[:, 1] - radii
    b4 = 1.0 - centers[:, 1] - radii
    
    # Overlap constraints (squared distance): dist^2 - (r_i + r_j)^2 >= 0
    # Using squared distances improves gradient behavior for optimizers
    dx = centers[:, 0, None] - centers[:, 0]
    dy = centers[:, 1, None] - centers[:, 1]
    dr = radii[:, None] + radii
    
    tri = np.tril_indices(N, -1)
    dist_sq = dx[tri]**2 + dy[tri]**2
    r_sum_sq = dr[tri]**2
    
    return np.concatenate([b1, b2, b3, b4, dist_sq - r_sum_sq])

def hex_init(seed):
    """Generate hexagonal lattice initialization with controlled noise."""
    np.random.seed(seed)
    r0 = 0.095
    centers = []
    y = r0
    row = 0
    while len(centers) < N:
        x = r0 if row % 2 == 0 else 2 * r0
        while x + r0 <= 1.0 and len(centers) < N:
            centers.append([x, y])
            x += 2 * r0
        y += np.sqrt(3) * r0
        row += 1
        
    while len(centers) < N:
        centers.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)])
        
    centers = np.array(centers[:N])
    radii = np.full(N, r0)
    
    # Add noise to break exact symmetry and help optimizer explore
    centers += np.random.normal(0, 0.008, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    return np.concatenate([centers.flatten(), radii])

def force_init(seed):
    """Generate initialization via repulsive force simulation."""
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (N, 2))
    radii = np.full(N, 0.08)
    
    # Relax positions to resolve overlaps
    for _ in range(500):
        forces = np.zeros((N, 2))
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.hypot(dx, dy)
                min_dist = radii[i] + radii[j] + 0.005
                if dist < min_dist and dist > 1e-6:
                    f = (min_dist - dist) * 0.15
                    fx, fy = f * dx / dist, f * dy / dist
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
        
        for i in range(N):
            r = radii[i]
            if centers[i, 0] < r + 0.02: forces[i, 0] += 0.08
            if centers[i, 0] > 1.0 - r - 0.02: forces[i, 0] -= 0.08
            if centers[i, 1] < r + 0.02: forces[i, 1] += 0.08
            if centers[i, 1] > 1.0 - r - 0.02: forces[i, 1] -= 0.08
            
        centers += forces * 0.6
        centers = np.clip(centers, 0.005, 0.995)
        
    for i in range(N):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
        
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Phase 1: Multiple diverse starts
    inits = [hex_init(s) for s in range(15)]
    inits += [force_init(s) for s in range(10)]
    
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_vars = res.x.copy()
        except Exception:
            pass
            
    if best_vars is None:
        best_vars = hex_init(0)
        
    # Phase 2: Local refinement to escape shallow local minima
    for _ in range(20):
        noisy = best_vars + np.random.normal(0, 1e-4, size=best_vars.shape)
        for i in range(N):
            r = max(0.0, noisy[2*N+i])
            noisy[2*N+i] = r
            noisy[i] = np.clip(noisy[i], r, 1.0-r)
            noisy[N+i] = np.clip(noisy[N+i], r, 1.0-r)
            
        try:
            res = minimize(objective, noisy, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_vars = res.x.copy()
        except Exception:
            pass
            
    centers = best_vars[:2*N].reshape(N, 2)
    radii = best_vars[2*N:]
    
    # Phase 3: Strict validity check and numerical repair
    valid = True
    for i in range(N):
        if radii[i] < 0:
            valid = False; break
        if centers[i, 0] - radii[i] < -1e-12 or centers[i, 0] + radii[i] > 1 + 1e-12:
            valid = False; break
        if centers[i, 1] - radii[i] < -1e-12 or centers[i, 1] + radii[i] > 1 + 1e-12:
            valid = False; break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i, 0]-centers[j, 0], centers[i, 1]-centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False; break
            if not valid: break
            
    if not valid:
        for _ in range(100):
            radii *= 0.998
            centers[:, 0] = np.clip(centers[:, 0], radii, 1.0-radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1.0-radii)
            ok = True
            for i in range(N):
                if centers[i, 0] + radii[i] > 1 + 1e-12 or centers[i, 0] - radii[i] < -1e-12:
                    ok = False; break
                if centers[i, 1] + radii[i] > 1 + 1e-12 or centers[i, 1] - radii[i] < -1e-12:
                    ok = False; break
            if ok:
                for i in range(N):
                    for j in range(i+1, N):
                        if np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1]) < radii[i]+radii[j]-1e-12:
                            ok = False; break
                    if not ok: break
            if ok: break
        best_sum = np.sum(radii)
        
    return centers, radii, float(best_sum)

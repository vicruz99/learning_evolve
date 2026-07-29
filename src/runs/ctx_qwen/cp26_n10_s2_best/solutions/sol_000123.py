# sol_000123 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000091 (state 364131c7) state=a0e9b088 sum of radii=2.625162 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap (squared)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(I_IDX))
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    c[4*N:] = dx**2 + dy**2 - (r[I_IDX] + r[J_IDX])**2
    return c

def relax_config(centers, radii, steps=150):
    """Force-directed relaxation to resolve overlaps and spread circles."""
    for _ in range(steps):
        forces = np.zeros_like(centers)
        for i in range(N):
            # Boundary repulsion
            if centers[i,0] - radii[i] < 0.01: forces[i,0] += 0.15
            if centers[i,0] + radii[i] > 0.99: forces[i,0] -= 0.15
            if centers[i,1] - radii[i] < 0.01: forces[i,1] += 0.15
            if centers[i,1] + radii[i] > 0.99: forces[i,1] -= 0.15
            
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                d = np.hypot(dx, dy)
                if d < radii[i] + radii[j] and d > 1e-8:
                    f = (radii[i] + radii[j] - d) * 3.0 / d
                    forces[i,0] += dx * f
                    forces[i,1] += dy * f
                    forces[j,0] -= dx * f
                    forces[j,1] -= dy * f
        centers += forces * 0.1
        centers = np.clip(centers, 0.02, 0.98)
    return centers

def get_feasible_r(centers):
    """Compute strictly feasible initial radii for given centers."""
    r = np.full(N, 0.5)
    for i in range(N):
        r[i] = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            val = d / 2.0
            if val < r[i]: r[i] = val
            if val < r[j]: r[j] = val
    return r * 0.75

def generate_initial_configs():
    """Generate a diverse set of initial configurations."""
    configs = []
    
    # 1. Hexagonal lattice patterns
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,6,7], [4,6,6,6,4], [5,5,6,5,5]]
    for pat in patterns:
        r0 = 0.09
        y = r0
        row_idx = 0
        pts = []
        for count in pat:
            x_start = r0 if row_idx % 2 == 0 else 2*r0
            for _ in range(count):
                pts.append([x_start, y])
                x_start += 2*r0
            y += r0 * np.sqrt(3)
            row_idx += 1
        if len(pts) >= N:
            configs.append(np.array(pts[:N]))
            
    # 2. Rotated hexagonal lattices
    if configs:
        base = configs[0]
        for ang in np.linspace(-0.25, 0.25, 6):
            c, s = np.cos(ang), np.sin(ang)
            p = base - 0.5
            p = p @ np.array([[c, -s], [s, c]]) + 0.5
            valid = (p[:,0]>=0.05) & (p[:,0]<=0.95) & (p[:,1]>=0.05) & (p[:,1]<=0.95)
            if np.sum(valid) >= N:
                configs.append(p[valid][:N])
                
    # 3. Grid and corner-cluster variations
    for _ in range(8):
        pts = np.random.uniform(0.08, 0.92, (N, 2))
        configs.append(pts)
        
    # 4. Corner-focused starts
    for _ in range(4):
        corners = np.array([[0.12,0.12], [0.88,0.12], [0.12,0.88], [0.88,0.88]])
        rest = np.random.uniform(0.25, 0.75, (N-4, 2))
        configs.append(np.vstack([corners, rest]))
        
    return configs

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    inits = generate_initial_configs()
    
    # Phase 1: Multi-start with pre-relaxation
    for centers in inits:
        r_init = get_feasible_r(centers)
        centers_relaxed = relax_config(centers.copy(), r_init, steps=100)
        r_relaxed = get_feasible_r(centers_relaxed)
        v0 = np.concatenate([centers_relaxed[:,0], centers_relaxed[:,1], r_relaxed])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-5:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        best_v = np.concatenate([inits[0][:,0], inits[0][:,1], np.full(N, 0.05)])
        
    # Phase 2: Perturbation & Refinement to escape local minima
    current_v = best_v.copy()
    for step in range(40):
        np.random.seed(step + 1000)
        v_p = current_v.copy()
        
        # Gradual shrink to unstick boundaries, then restore during optimization
        shrink = 0.995 - step * 0.0005
        v_p[2*N:] *= max(0.95, shrink)
        
        # Controlled center perturbation
        noise_mag = 0.008 * (1.0 - step/50.0)
        v_p[:2*N] += np.random.uniform(-noise_mag, noise_mag, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
        
        # Quick relaxation to ensure feasibility after perturbation
        c_p = v_p[:2*N].reshape(N, 2)
        r_p = v_p[2*N:]
        c_p_relaxed = relax_config(c_p, r_p, steps=50)
        v_p[:2*N] = c_p_relaxed.flatten()
        v_p[2*N:] = get_feasible_r(c_p_relaxed)
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-5:
                    best_sum = curr_sum
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack((cx, cy))
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        mr = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        cr[i] = min(cr[i], mr)
        
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, cr, float(np.sum(cr))

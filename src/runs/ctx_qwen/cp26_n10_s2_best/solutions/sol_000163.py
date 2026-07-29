# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000086 (state e307a773) state=b9c973b0 sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and squared non-overlap distances."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
        (x[PAIR_I] - x[PAIR_J])**2 + (y[PAIR_I] - y[PAIR_J])**2 - (r[PAIR_I] + r[PAIR_J])**2
    ])
    return c

def get_initial_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.minimum(np.minimum(centers[:,0], 1.0-centers[:,0]), 
                   np.minimum(centers[:,1], 1.0-centers[:,1]))
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            val = d / 2.0
            r[i] = min(r[i], val)
            r[j] = min(r[j], val)
    return r * 0.5

def generate_configs():
    """Generate diverse initial configurations."""
    configs = []
    # 1. Hexagonal lattices with various rotations
    for r0 in np.linspace(0.07, 0.11, 9):
        pts = []
        y = r0
        row = 0
        while len(pts) < N+10:
            xs = r0 if row%2==0 else 2*r0
            x = xs
            while x <= 1.0-r0 and len(pts) < N+10:
                pts.append([x, y])
                x += 2*r0
            y += np.sqrt(3)*r0
            row += 1
        pts = np.array(pts[:N])
        for ang in [0, 0.05, -0.05, 0.1, -0.1]:
            c, s = np.cos(ang), np.sin(ang)
            rot = pts - 0.5
            rot = rot @ np.array([[c, -s], [s, c]]) + 0.5
            valid = (rot[:,0]>=0.02) & (rot[:,0]<=0.98) & (rot[:,1]>=0.02) & (rot[:,1]<=0.98)
            if np.sum(valid) >= N:
                configs.append(rot[valid][:N])
                
    # 2. Grid configurations
    for s in np.linspace(0.15, 0.22, 8):
        pts = np.array([[i*s+0.05, j*s+0.05] for i in range(6) for j in range(5)])[:N]
        configs.append(pts)
        
    # 3. Force-relaxed random configurations
    for seed in range(25):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        for _ in range(100):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 0.3 and d > 1e-6:
                        f = (0.3 - d) * 0.5 / d
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts)
        
    return configs

def run_packing():
    np.random.seed(42)
    configs = generate_configs()
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Phase 1: Multi-start optimization
    for c_init in configs:
        r_init = get_initial_radii(c_init)
        v0 = np.concatenate([c_init[:,0], c_init[:,1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-8:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        return np.zeros((N,2)), np.zeros(N), 0.0
        
    # Phase 2: Iterative perturbation & refinement to escape local minima
    current_v = best_v.copy()
    for step in range(40):
        np.random.seed(step + 3000)
        v_pert = current_v.copy()
        noise_scale = 0.008 - step * 0.0001
        v_pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        # Progressively shrink radii to allow circles to rearrange
        shrink_factor = 0.85 - step * 0.002
        v_pert[2*N:] *= max(0.6, shrink_factor)
        
        # Recompute feasible radii from perturbed centers
        c_pts = v_pert[:2*N].reshape(N, 2)
        v_pert[2*N:] = get_initial_radii(c_pts) * 0.85
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-8:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            pass
            
    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validator compliance
    radii = np.minimum(radii, np.minimum(centers[:,0], 1.0-centers[:,0]))
    radii = np.minimum(radii, np.minimum(centers[:,1], 1.0-centers[:,1]))
    
    for _ in range(20):
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    
    return centers, radii, float(np.sum(radii))

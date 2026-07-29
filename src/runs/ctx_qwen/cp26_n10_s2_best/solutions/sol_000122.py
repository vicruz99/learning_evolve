# sol_000122 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000091 (state 364131c7) state=12f22b36 sum of radii=2.628402 correctness=1.0
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
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Pre-allocate constraint array
    c = np.empty(4*N + len(PAIR_I))
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap constraints (squared for numerical stability)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def generate_initial_configs():
    """Generate diverse initial center configurations."""
    configs = []
    
    # 1. Hexagonal lattice variations with different densities and shifts
    for r0 in np.linspace(0.085, 0.110, 6):
        for shift in np.linspace(-0.02, 0.02, 3):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 10:
                x_start = r0 + shift + (row % 2) * r0
                x = x_start
                while x <= 1.0 - r0 and len(pts) < N + 10:
                    pts.append([x, y])
                    x += 2 * r0
                y += np.sqrt(3) * r0
                row += 1
            pts = np.array(pts[:N])
            configs.append(pts)
            
    # 2. Square grid variations
    for s in np.linspace(0.14, 0.20, 5):
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([s + i*s, s + j*s])
        pts = np.array(pts[:N])
        configs.append(pts)
        
    # 3. Force-repelled random starts to find organic dense packings
    for seed in range(25):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        for _ in range(80):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
                    if d < 0.30 and d > 1e-4:
                        f = (0.30 - d) * 0.6 / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.04
            pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts)
        
    return configs

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    configs = generate_initial_configs()
    
    # Phase 1: Multi-start optimization from diverse basins
    for cfg in configs:
        r_init = np.full(N, 0.04)
        v0 = np.concatenate([cfg[:,0], cfg[:,1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-7 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local refinement with decaying perturbation to escape local minima
    if best_v is not None:
        current_v = best_v
        for step in range(40):
            noise_scale = 0.006 * np.exp(-step * 0.08)
            pert = current_v.copy()
            pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            pert[2*N:] *= 0.985  # Shrink to ensure feasibility after perturbation
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                s = -res.fun
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7 and s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v
            except Exception:
                pass
                
    # Extract optimal configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack((cx, cy))
    
    # Post-processing: strict boundary enforcement
    cr = np.minimum(cr, np.minimum(centers[:,0], 1.0 - centers[:,0]))
    cr = np.minimum(cr, np.minimum(centers[:,1], 1.0 - centers[:,1]))
    
    # Post-processing: strict non-overlap enforcement iteratively
    for _ in range(15):
        dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
        dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
        dist = np.sqrt(dx**2 + dy**2)
        sum_r = cr[PAIR_I] + cr[PAIR_J]
        excess = sum_r - dist
        if np.max(excess) < 1e-9:
            break
        shrink = np.maximum(0.0, excess) / 2.0 + 1e-9
        cr[PAIR_I] = np.maximum(0.0, cr[PAIR_I] - shrink)
        cr[PAIR_J] = np.maximum(0.0, cr[PAIR_J] - shrink)
        
    # Final safety scale to guarantee validator's 1e-12 tolerance is met
    cr *= 0.999998
    
    return centers, cr, float(np.sum(cr))

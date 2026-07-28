# sol_000076 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 81a0d5f4) state=7945f8b3 sum of radii=2.501494 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars_arr, n):
    """
    Computes inequality constraints for equal-radius packing.
    All constraints must be >= 0 for SLSQP.
    """
    x = vars_arr[0:-1:2]
    y = vars_arr[1:-1:2]
    r = vars_arr[-1]
    
    # Pairwise squared distances
    xs = x[:, None]
    ys = y[:, None]
    dx = xs - xs.T
    dy = ys - ys.T
    dist_sq = dx**2 + dy**2
    idx = np.triu_indices(n, k=1)
    
    # Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    # Overlap: dist_sq >= 4*r^2
    return np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
        dist_sq[idx] - 4.0 * r**2
    ])

def objective_func(vars_arr):
    """Objective: minimize negative radius => maximize radius"""
    return -vars_arr[-1]

def force_simulate(centers, radii, steps=1500):
    """
    Force-directed relaxation to pack circles tightly and increase radius gradually.
    """
    n = len(radii)
    vel = np.zeros_like(centers)
    dt = 0.005
    damping = 0.90
    k_rep = 50.0
    k_wall = 20.0
    
    for _ in range(steps):
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                req = radii[i] + radii[j]
                if dist < req and dist > 1e-9:
                    overlap = req - dist
                    f = k_rep * overlap / dist
                    forces[i] += f * diff
                    forces[j] -= f * diff
                    
            # Wall repulsion
            if centers[i, 0] - radii[i] < 0:
                forces[i, 0] += k_wall * (radii[i] - centers[i, 0])
            if centers[i, 0] + radii[i] > 1.0:
                forces[i, 0] -= k_wall * (centers[i, 0] + radii[i] - 1.0)
            if centers[i, 1] - radii[i] < 0:
                forces[i, 1] += k_wall * (radii[i] - centers[i, 1])
            if centers[i, 1] + radii[i] > 1.0:
                forces[i, 1] -= k_wall * (centers[i, 1] + radii[i] - 1.0)
                
        vel = damping * vel + forces * dt
        centers += vel
        centers = np.clip(centers, 0.0, 1.0)
        
        # Gradually expand radii to drive packing tighter
        radii *= 1.00008
        
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)
    
    best_r = 0.0
    best_centers = None
    
    # --- Generate Diverse Initial Configurations ---
    configs = []
    
    # 1. Base Hexagonal Lattice
    pts = []
    r_init = 0.08
    y = r_init
    row = 0
    while len(pts) < n:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
    configs.append(np.array(pts[:n]))
    
    # 2. Perturbed Hexagonal Configs
    for _ in range(3):
        cfg = configs[0].copy()
        cfg += np.random.normal(0, 0.02, cfg.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        configs.append(cfg)
        
    # 3. Grid-based Config
    grid = np.array([(i*0.19+0.1, j*0.19+0.1) for i in range(6) for j in range(5)]).reshape(-1, 2)
    configs.append(grid[:n])
    
    # Bounds: centers in [0,1], radius in [0.01, 0.2]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.2)]
    
    # --- Optimization Phase ---
    for cfg in configs:
        # Phase 1: Force simulation to find a tight, valid arrangement
        sim_centers, sim_radii = force_simulate(cfg.copy(), np.full(n, 0.08))
        
        # Phase 2: Precise SLSQP optimization
        x0 = np.zeros(2 * n + 1)
        x0[0:-1:2] = sim_centers[:, 0]
        x0[1:-1:2] = sim_centers[:, 1]
        x0[-1] = 0.09  # Start with a feasible radius
        
        try:
            res = minimize(
                objective_func, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': compute_constraints, 'args': (n,)},
                options={'maxiter': 5000, 'ftol': 1e-12}
            )
            if res.x[-1] > best_r:
                best_r = res.x[-1]
                best_centers = res.x[:-1].reshape(n, 2).copy()
        except Exception:
            continue
            
    # --- Refinement Phase ---
    # Try escaping local minima by perturbing the best result found so far
    for _ in range(4):
        if best_centers is None:
            break
        pert = best_centers + np.random.normal(0, 0.003, best_centers.shape)
        pert = np.clip(pert, 0.01, 0.99)
        x0 = np.zeros(2 * n + 1)
        x0[0:-1:2] = pert[:, 0]
        x0[1:-1:2] = pert[:, 1]
        x0[-1] = best_r * 0.99
        
        try:
            res = minimize(
                objective_func, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': compute_constraints, 'args': (n,)},
                options={'maxiter': 3000, 'ftol': 1e-12}
            )
            if res.x[-1] > best_r:
                best_r = res.x[-1]
                best_centers = res.x[:-1].reshape(n, 2).copy()
        except Exception:
            pass
            
    # Safety fallback
    if best_centers is None:
        r_fb = 0.095
        best_centers = np.array([(i * 2 * r_fb + r_fb, j * 2 * r_fb + r_fb) 
                                 for j in range(5) for i in range(5)] + [[0.55, 0.55]])
        best_r = r_fb
        
    # Apply tiny safety margin to guarantee strict validity against numerical tolerance
    radii = np.full(n, best_r * 0.9999999)
    sum_r = float(np.sum(radii))
    
    return best_centers, radii, sum_r

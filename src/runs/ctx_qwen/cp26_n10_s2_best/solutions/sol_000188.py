# sol_000188 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000144 (state c0d23801) state=f07b18d4 sum of radii=2.628596 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints_joint(v):
    """Inequality constraints: boundaries and squared non-overlap."""
    x, y, r = v[:N], v[N:2*N], v[2*N:]
    c = np.empty(4*N + len(PAIR_I))
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dr = r[PAIR_I] + r[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - dr**2
    return c

def get_feasible_radii(centers, scale=0.95):
    """Compute strictly feasible initial radii based on local geometry."""
    x, y = centers[:, 0], centers[:, 1]
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    r = np.minimum(wall, min_dists / 2.0)
    return r * scale

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    for r0 in np.linspace(0.08, 0.12, 5):
        for angle in np.linspace(-0.3, 0.3, 5):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 5:
                xs = r0 if row % 2 == 0 else 2 * r0
                x = xs
                while x <= 1.0 - r0 and len(pts) < N + 5:
                    pts.append([x, y])
                    x += 2 * r0
                y += np.sqrt(3) * r0
                row += 1
            pts = np.array(pts[:N])
            c, s = np.cos(angle), np.sin(angle)
            pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
            pts = np.clip(pts, 0.01, 0.99)
            configs.append(pts)
    return configs

def obj_k(vk):
    """Objective for coordinate-wise refinement."""
    return -vk[2]

def cons_k(vk, other_centers, other_radii):
    """Constraints for coordinate-wise refinement."""
    xk, yk, rk = vk
    c = np.array([xk - rk, 1.0 - xk - rk, yk - rk, 1.0 - yk - rk])
    dx = xk - other_centers[:, 0]
    dy = yk - other_centers[:, 1]
    dists = np.hypot(dx, dy)
    c = np.concatenate([c, dists - (rk + other_radii)])
    return c

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_joint}
    
    best_v = None
    best_sum = -1.0
    
    # Generate diverse initial configurations
    inits = generate_initial_configs()
    for seed in range(10):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        # Quick force relaxation to spread points
        for _ in range(40):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i]-pts[j])
                    if d < 0.25 and d > 1e-5:
                        f = (0.25 - d)/d
                        diff = pts[i]-pts[j]
                        forces[i] += f*diff
                        forces[j] -= f*diff
            pts += forces * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)
        
    # Phase 1: Multi-start joint optimization
    for centers in inits:
        r_init = get_feasible_radii(centers, 0.85)
        v0 = np.concatenate([centers[:,0], centers[:,1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum and np.min(constraints_joint(res.x)) >= -1e-7:
                best_sum = s
                best_v = res.x.copy()
        except:
            pass
            
    if best_v is None:
        return np.zeros((N,2)), np.zeros(N), 0.0
        
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 2: Perturbation & Re-optimization to escape local minima
    for step in range(20):
        noise_scale = 0.003 * (1.0 - step/20.0)
        c_pert = centers + np.random.uniform(-noise_scale, noise_scale, centers.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        r_pert = get_feasible_radii(c_pert, 0.90)
        v_pert = np.concatenate([c_pert[:,0], c_pert[:,1], r_pert])
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum and np.min(constraints_joint(res.x)) >= -1e-7:
                best_sum = s
                best_v = res.x.copy()
                centers = np.column_stack((best_v[:N], best_v[N:2*N]))
                radii = best_v[2*N:].copy()
        except:
            pass
            
    # Phase 3: Coordinate-wise local refinement
    # Optimizes each circle sequentially while fixing others, effectively untangling constraints
    for _ in range(5):
        for k in range(N):
            other_centers = np.delete(centers, k, axis=0)
            other_radii = np.delete(radii, k)
            
            vk0 = [centers[k,0], centers[k,1], radii[k]]
            bds_k = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)]
            cons_k_dict = {'type': 'ineq', 'fun': cons_k, 'args': (other_centers, other_radii)}
            try:
                res_k = minimize(obj_k, vk0, method='SLSQP', bounds=bds_k, constraints=cons_k_dict,
                                 options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
                if -res_k.fun > radii[k]:
                    centers[k] = res_k.x[:2]
                    radii[k] = res_k.x[2]
            except:
                pass
                
    # Phase 4: Final joint polish
    v_final = np.concatenate([centers[:,0], centers[:,1], radii])
    try:
        res_final = minimize(objective, v_final, method='SLSQP', bounds=bounds, constraints=cons,
                             options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
        if -res_final.fun > best_sum:
            best_sum = -res_final.fun
            best_v = res_final.x.copy()
            centers = np.column_stack((best_v[:N], best_v[N:2*N]))
            radii = best_v[2*N:].copy()
    except:
        pass
        
    # Strict post-processing for validator compliance
    radii = np.minimum(radii, np.minimum(centers[:,0], 1.0-centers[:,0]))
    radii = np.minimum(radii, np.minimum(centers[:,1], 1.0-centers[:,1]))
    radii = np.maximum(radii, 0.0)
    
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))

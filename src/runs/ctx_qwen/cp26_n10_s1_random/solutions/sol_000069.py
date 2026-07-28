# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000052 (state e51e4326) state=b115254f sum of radii=2.505396 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj(x, n):
    """Objective: minimize negative radius (maximize r)"""
    return -x[-1]

def cons(x, n):
    """Inequality constraints: boundary distances and pairwise distances >= 2r"""
    c = x[:2*n].reshape(n, 2)
    r = x[-1]
    
    # Boundary constraints: r <= coord <= 1-r
    b = np.concatenate([
        c[:, 0] - r, 1.0 - c[:, 0] - r,
        c[:, 1] - r, 1.0 - c[:, 1] - r
    ])
    
    # Pairwise non-overlap: ||c_i - c_j||^2 >= (2r)^2
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    idx = np.triu_indices(n, k=1)
    p = dist_sq[idx] - 4.0 * r**2
    
    return np.concatenate([b, p])

def simulate(c, r, n, steps=2000):
    """Force-directed repulsion simulation to spread circles evenly"""
    c = c.copy()
    for _ in range(steps):
        forces = np.zeros_like(c)
        
        # Boundary repulsion
        forces[:, 0] += np.clip(r - c[:, 0], 0, None) * 100.0
        forces[:, 0] -= np.clip(c[:, 0] - (1.0 - r), 0, None) * 100.0
        forces[:, 1] += np.clip(r - c[:, 1], 0, None) * 100.0
        forces[:, 1] -= np.clip(c[:, 1] - (1.0 - r), 0, None) * 100.0
        
        # Pairwise repulsion
        diff = c[:, None, :] - c[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_d = 2.0 * r
        overlap = np.maximum(0.0, min_d - dists)
        rep = (overlap * 50.0) / dists
        
        fx = np.sum(diff[:, :, 0] * rep, axis=1)
        fy = np.sum(diff[:, :, 1] * rep, axis=1)
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        c += forces * 0.005
        c = np.clip(c, r, 1.0 - r)
    return c

def get_hex(r_init, n):
    """Generate hexagonal lattice initialization"""
    pts = []
    y = r_init
    row = 0
    while len(pts) < n:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r_init
        y += r_init * np.sqrt(3)
        row += 1
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_centers = None
    best_r = 0.0
    best_sum = 0.0
    
    np.random.seed(42)
    base_hex = get_hex(0.095, n)
    configs = [base_hex]
    
    # Generate perturbed initial configurations
    for _ in range(10):
        cfg = base_hex + np.random.uniform(-0.025, 0.025, base_hex.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        configs.append(cfg)

    bounds = [(0.0, 1.0)] * (2*n) + [(0.09, 0.105)]
    
    for init_c in configs:
        # Pre-spread circles using simulation
        sim_c = simulate(init_c, 0.098, n, steps=2500)
        x0 = np.concatenate([sim_c.flatten(), [0.098]])
        
        try:
            res = minimize(
                obj, x0, args=(n,), method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': cons, 'args': (n,)},
                options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False}
            )
            
            if res.success or res.fun < -0.099:
                r_cand = res.x[-1]
                c_val = cons(res.x, n)
                # Accept if constraints are satisfied within tolerance
                if np.min(c_val) >= -1e-5:
                    if r_cand > best_r:
                        best_r = r_cand
                        best_centers = res.x[:2*n].reshape(n, 2)
                        best_sum = r_cand * n
        except Exception:
            continue

    # Fallback if optimization fails unexpectedly
    if best_centers is None:
        best_centers = base_hex
        best_r = 0.095
        best_sum = best_r * n

    # Post-process: compute exact maximum feasible equal radius for the optimized centers
    c = best_centers
    min_wall = np.min(np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                                 np.minimum(c[:, 1], 1.0 - c[:, 1])))
    
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    true_r = min(min_wall, min_pair)
    if true_r > best_r:
        best_r = true_r
        best_sum = best_r * n
        
    # Apply tiny margin to strictly satisfy validator tolerances
    final_r = best_r * 0.99999
    radii = np.full(n, final_r)
    
    return best_centers, radii, float(np.sum(radii))

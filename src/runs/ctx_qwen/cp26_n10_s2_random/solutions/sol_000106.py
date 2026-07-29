# sol_000106 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000076 (state b16097a6) state=b73c9897 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, k=1)

def objective(v):
    """Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute boundary and non-overlap constraints. All values must be >= 0."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = np.empty(4*N + N*(N-1)//2)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con[:N] = c[:, 0] - r
    con[N:2*N] = 1.0 - c[:, 0] - r
    con[2*N:3*N] = c[:, 1] - r
    con[3*N:4*N] = 1.0 - c[:, 1] - r
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    c_i = c[TRIU_I]
    c_j = c[TRIU_J]
    d = np.sqrt(np.sum((c_i - c_j)**2, axis=1))
    con[4*N:] = d - (r[TRIU_I] + r[TRIU_J])
    return con

def get_init_radii(c):
    """Compute safe initial radii for a given set of centers."""
    rb = np.minimum(np.minimum(c[:,0], 1.0-c[:,0]), np.minimum(c[:,1], 1.0-c[:,1]))
    dists = np.sqrt(np.sum((c[:,None,:] - c[None,:,:])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    rp = 0.5 * np.min(dists, axis=1)
    return np.maximum(np.minimum(rb, rp) * 0.95, 1e-5)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    starts = []
    
    # 1. Hexagonal lattice patterns with diverse row structures
    patterns = [[5,5,5,5,6], [5,6,5,6,4], [6,5,6,5,4], [4,6,6,6,4], 
                [5,5,6,5,5], [6,4,6,5,5], [5,4,6,6,5], [4,5,6,5,6],
                [5,5,5,6,5], [6,5,5,6,4], [5,6,6,5,4], [4,5,5,6,6],
                [5,6,5,5,5], [6,5,5,5,5], [5,5,4,6,6], [6,4,5,6,5]]
                
    for pat in patterns:
        c = []
        r_est = 0.098
        y = r_est
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                c.append([x, y])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3)
        c = np.array(c[:N])
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.12, 0.88)
        starts.append(c)
        
    # 2. Random starts with force-directed repulsion to ensure initial spread
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(150):
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.18:
                        push = (0.18 - d) * 0.5
                        c[i] += d_vec/d * push
                        c[j] -= d_vec/d * push
        c = np.clip(c, 0.1, 0.9)
        starts.append(c)
        
    # Phase 1: Multi-start constrained optimization
    for c_init in starts:
        r_init = get_init_radii(c_init)
        v0 = np.concatenate([c_init.flatten(), r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 8000, 'ftol': 1e-14})
            c_val = constraints(res.x)
            if np.min(c_val) > -1e-6:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = np.concatenate([starts[0].flatten(), get_init_radii(starts[0])])
        
    # Phase 2: Iterative perturbation to escape local minima
    for step in range(30):
        noise = 0.004 * (0.85**step)
        v_pert = best_v + rng.normal(0, noise, best_v.shape)
        v_pert = np.clip(v_pert, 0.0, 1.0)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.0, 0.5)
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 5000, 'ftol': 1e-14})
            if np.min(constraints(res.x)) > -1e-6:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Extract results
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict numerical repair to guarantee validation passes
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i]-centers[j])
                req = radii[i]+radii[j]
                if d < req - 1e-12:
                    shrink = (req - d)/2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            x,y,r = centers[i,0], centers[i,1], radii[i]
            mr = min(x, 1.0-x, y, 1.0-y)
            if r > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))

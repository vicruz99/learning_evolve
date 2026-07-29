# sol_000092 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000068 (state 22e68fa8) state=baf0527b sum of radii=2.619708 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: minimize negative sum of radii to maximize total radius."""
    return -np.sum(v[2::3])

def constraints(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    idx = np.triu_indices(N, 1)
    dx = x[idx[0]] - x[idx[1]]
    dy = y[idx[0]] - y[idx[1]]
    dr = r[idx[0]] + r[idx[1]]
    
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def generate_inits():
    """Generates a diverse set of initial configurations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Hexagonal grids with different row patterns summing to 26
    row_patterns = [
        [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], [5, 5, 6, 5, 5], 
        [5, 5, 5, 6, 5], [5, 5, 5, 5, 6], [4, 6, 6, 6, 4],
        [5, 5, 5, 5, 4, 2], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [4, 5, 5, 5, 5, 2], [6, 4, 6, 5, 5]
    ]
    
    for pat in row_patterns:
        centers = []
        r0 = 0.09
        y = r0
        for row_idx, count in enumerate(pat):
            x_start = r0 if row_idx % 2 == 0 else 2*r0
            x = x_start
            for _ in range(count):
                centers.append([x, y])
                x += 2*r0
            y += np.sqrt(3)*r0
        centers = np.array(centers[:N])
        # Perturb to break symmetry
        centers += rng.normal(0, 0.005, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        rs = np.full(N, 0.06)
        v = np.zeros(3*N)
        v[0::3] = centers[:, 0]
        v[1::3] = centers[:, 1]
        v[2::3] = rs
        inits.append(v)
        
    # Random dense starts
    for _ in range(25):
        c = rng.random((N, 2)) * 0.8 + 0.1
        v = np.zeros(3*N)
        v[0::3] = c[:, 0]
        v[1::3] = c[:, 1]
        v[2::3] = 0.05
        inits.append(v)
        
    # Corner-focused starts
    for _ in range(10):
        c = rng.random((N, 2)) * 0.6 + 0.2
        c[:4] = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        v = np.zeros(3*N)
        v[0::3] = c[:, 0]
        v[1::3] = c[:, 1]
        v[2::3] = 0.06
        inits.append(v)
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Main function to pack 26 circles in a unit square."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    inits = generate_inits()
    
    best_v = None
    best_sum = -np.inf
    
    # Phase 1: Multi-start optimization
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13})
            if np.min(constraints(res.x)) >= -1e-7:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation search to escape local minima
    if best_v is not None:
        rng = np.random.default_rng(123)
        for _ in range(50):
            v_try = best_v + rng.normal(0, 0.0015, best_v.shape)
            v_try[0::3] = np.clip(v_try[0::3], 0.02, 0.98)
            v_try[1::3] = np.clip(v_try[1::3], 0.02, 0.98)
            v_try[2::3] = np.clip(v_try[2::3], 0.01, 0.45)
            try:
                res = minimize(objective, v_try, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
                if np.min(constraints(res.x)) >= -1e-7:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_v = res.x.copy()
            except Exception:
                pass
                
    if best_v is None:
        best_v = inits[0]
        
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = best_v[2::3].copy()
    
    # Phase 3: Iterative Radius Expansion to push beyond local optima
    for _ in range(15):
        radii *= 1.0005
        v_exp = np.zeros(3*N)
        v_exp[0::3] = centers[:, 0]
        v_exp[1::3] = centers[:, 1]
        v_exp[2::3] = radii
        try:
            res = minimize(objective, v_exp, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
            if np.min(constraints(res.x)) >= -1e-7:
                centers = np.column_stack((res.x[0::3], res.x[1::3]))
                radii = res.x[2::3]
        except Exception:
            break
            
    # Phase 4: Deterministic Repair for Strict Validation Compliance
    for _ in range(50):
        changed = False
        # Resolve pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Clamp to boundaries
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-9:
                radii[i] = mr
                changed = True
                
        if not changed:
            break
            
    # Ensure non-negativity
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))

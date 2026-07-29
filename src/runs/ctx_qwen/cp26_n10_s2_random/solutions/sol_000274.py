# sol_000274 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000237 (state 963256f0) state=587b4694 sum of radii=2.603511 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute constant LP constraint matrix structure for pairwise distances
# Constraint: r_i + r_j <= dist(i, j)
A_LP = np.zeros((N * (N - 1) // 2, N))
PAIR_INDICES = []
_lp_row = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[_lp_row, i] = 1.0
        A_LP[_lp_row, j] = 1.0
        PAIR_INDICES.append((i, j))
        _lp_row += 1

def solve_lp_and_grad(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns radii, sum_radii, and gradient w.r.t centers.
    """
    c = np.clip(centers, 0.001, 0.999)
    ub = np.minimum(
        np.minimum(c[:, 0], 1.0 - c[:, 0]),
        np.minimum(c[:, 1], 1.0 - c[:, 1])
    )
    ub = np.maximum(ub, 1e-7)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[np.triu_indices(N, k=1)]
    
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    
    # Extract dual marginals safely
    duals = np.zeros(len(PAIR_INDICES))
    if hasattr(res, 'marginals') and res.marginals is not None:
        if hasattr(res.marginals, 'ineqlin'):
            duals = res.marginals.ineqlin
        elif hasattr(res.marginals, 'ineq'):
            duals = res.marginals.ineq
    elif hasattr(res, 'ineqlin'):
        duals = res.ineqlin.marginals if hasattr(res.ineqlin, 'marginals') else 0
        
    grad = np.zeros_like(c)
    idx = 0
    for i, j in PAIR_INDICES:
        mu = duals[idx]
        if mu > 1e-10:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    # Boundary gradient contributions
    for i in range(N):
        xi, yi = c[i]
        ri = radii[i]
        if ri + 1e-8 >= xi: grad[i, 0] += 1.0
        if ri + 1e-8 >= 1.0 - xi: grad[i, 0] -= 1.0
        if ri + 1e-8 >= yi: grad[i, 1] += 1.0
        if ri + 1e-8 >= 1.0 - yi: grad[i, 1] -= 1.0
        
    return radii, np.sum(radii), grad

def obj_func(x):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    c = x.reshape(N, 2)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def generate_starts(rng):
    """Generates diverse starting configurations."""
    starts = []
    
    # 1. Hexagonal lattices with various rotations
    for angle in [0.0, 0.05, 0.12, 0.18, 0.25]:
        pts = []
        r0 = 0.10
        y = r0
        row = 0
        while len(pts) < N:
            x = r0 + (row % 2) * r0
            while x + r0 < 1.0 and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:N])
        if angle != 0.0:
            cx, cy = 0.5, 0.5
            pts -= [cx, cy]
            ca, sa = np.cos(angle), np.sin(angle)
            pts = pts @ [[ca, -sa], [sa, ca]]
            pts += [cx, cy]
            pts = np.clip(pts, 0.05, 0.95)
        pts += rng.normal(0, 0.003, pts.shape)
        starts.append(pts)
        
    # 2. Square grid
    gs = 5
    gx = np.linspace(0.12, 0.88, gs)
    gy = np.linspace(0.12, 0.88, gs)
    cx, cy = np.meshgrid(gx, gy)
    grid = np.column_stack([cx.flatten(), cy.flatten()])
    grid = np.vstack([grid, [0.5, 0.5]])
    starts.append(grid + rng.normal(0, 0.002, grid.shape))
    
    # 3. Corner-heavy starts
    for _ in range(5):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        starts.append(c)
        
    # 4. Force-directed repelled starts
    for _ in range(5):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(300):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    v = c[i] - c[j]
                    d = np.linalg.norm(v) + 1e-6
                    if d < 0.15:
                        p = (0.15 - d) / (d**2)
                        f[i] += v * p
                        f[j] -= v * p
            c += f * 0.01
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def optimize_lbfgs(c0):
    """Runs L-BFGS-B optimization on centers using exact LP gradient."""
    c = c0.copy()
    res = minimize(obj_func, c.flatten(), jac=True, method='L-BFGS-B',
                   bounds=[(0.001, 0.999)] * (2 * N),
                   options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
    return res.x.reshape(N, 2), -res.fun

def optimize_sa(c0, rng):
    """Simulated annealing on centers to escape local minima."""
    c = c0.copy()
    radii, s_curr, _ = solve_lp_and_grad(c)
    best_c, best_s = c.copy(), s_curr
    T = 0.006
    for step in range(800):
        T *= 0.996
        c_new = c + rng.normal(0, T, c.shape)
        c_new = np.clip(c_new, 0.005, 0.995)
        radii_new, s_new, _ = solve_lp_and_grad(c_new)
        delta = s_new - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T * 5.0, 1e-8)):
            c = c_new
            s_curr = s_new
            if s_new > best_s:
                best_s = s_new
                best_c = c.copy()
    return best_c, best_s

def optimize_joint(c0, r0):
    """Joint SLSQP refinement for precise constraint satisfaction."""
    n = N
    v0 = np.concatenate([c0.flatten(), r0])
    
    def obj(v):
        return -np.sum(v[2 * n:])
        
    def cons(v):
        cc = v[:2 * n].reshape(n, 2)
        rr = v[2 * n:]
        c_list = [
            cc[:, 0] - rr,
            1.0 - cc[:, 0] - rr,
            cc[:, 1] - rr,
            1.0 - cc[:, 1] - rr
        ]
        idx = np.triu_indices(n, 1)
        d = np.linalg.norm(cc[idx[0]] - cc[idx[1]], axis=1)
        c_list.append(d - (rr[idx[0]] + rr[idx[1]]))
        return np.concatenate(c_list)
        
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons(res.x)) >= -1e-9:
            return res.x[:2 * n].reshape(n, 2), res.x[2 * n:], np.sum(res.x[2 * n:])
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def repair_packing(centers, radii):
    """Deterministically shrinks radii to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(60):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    starts = generate_starts(rng)
    
    best_c, best_r, best_s = None, None, -1.0
    candidates = []
    
    # Phase 1: L-BFGS-B on all starts
    for s in starts:
        c, s_val = optimize_lbfgs(s)
        r, _, _ = solve_lp_and_grad(c)
        candidates.append((c, r, s_val))
        
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Keep top 3 for intensive polishing
    for c, r, s in candidates[:3]:
        if s > best_s:
            best_s = s
            best_c = c
            best_r = r
            
    # Phase 2: SA on the best candidate
    c_sa, s_sa = optimize_sa(best_c, rng)
    r_sa, _, _ = solve_lp_and_grad(c_sa)
    if s_sa > best_s:
        best_s = s_sa
        best_c = c_sa
        best_r = r_sa
        
    # Phase 3: Joint SLSQP polish
    c_j, r_j, s_j = optimize_joint(best_c, best_r)
    if s_j > best_s:
        best_s = s_j
        best_c = c_j
        best_r = r_j
        
    # Phase 4: Repair
    best_r = repair_packing(best_c, best_r)
    final_sum = float(np.sum(best_r))
    
    return best_c, best_r, final_sum

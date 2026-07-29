import numpy as np
from scipy.optimize import linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)
A_LP = np.zeros((NUM_PAIRS, N))
A_LP[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_LP[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 1e-4)

def evaluate_sum_radii(centers):
    """Objective function: returns sum of optimal radii for given centers."""
    r = solve_lp_radii(centers)
    return np.sum(r)

def boundary_push(centers, rng):
    """Heuristically push circles towards boundaries/corners to maximize clearance."""
    c = centers.copy()
    best_s = evaluate_sum_radii(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            for dx, dy in [(0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02),
                           (0.02, 0.02), (-0.02, 0.02), (0.02, -0.02), (-0.02, -0.02)]:
                c_new = c.copy()
                c_new[i, 0] = np.clip(c_new[i, 0] + dx, 0.01, 0.99)
                c_new[i, 1] = np.clip(c_new[i, 1] + dy, 0.01, 0.99)
                s_new = evaluate_sum_radii(c_new)
                if s_new > best_s + 1e-7:
                    c = c_new
                    best_s = s_new
                    improved = True
    return c, best_s

def sa_optimize(c0, rng, steps=3000):
    """Simulated annealing on center coordinates."""
    c = c0.copy()
    best_sum = evaluate_sum_radii(c)
    best_c = c.copy()
    temp = 0.004
    
    for step in range(steps):
        scale = max(0.0002, 0.004 * np.exp(-step / 800.0))
        n_pert = rng.integers(1, 4)
        idxs = rng.choice(N, n_pert, replace=False)
        c_pert = c.copy()
        c_pert[idxs] += rng.normal(0, scale, (n_pert, 2))
        c_pert = np.clip(c_pert, 0.005, 0.995)
        
        s_pert = evaluate_sum_radii(c_pert)
        
        if s_pert > best_sum:
            c = c_pert
            best_sum = s_pert
            best_c = c_pert.copy()
            temp = 0.004
        elif rng.random() < np.exp((s_pert - best_sum) / max(temp, 1e-6)):
            c = c_pert
            best_sum = s_pert
            
        temp *= 0.999
    return best_c, best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    best_c = None
    best_s = 0.0
    
    # Generate diverse initial configurations
    inits = []
    # Hexagonal lattices with varying spacings
    for sp in np.linspace(0.16, 0.24, 8):
        c = np.zeros((N, 2))
        idx = 0
        y = 0.05
        row = 0
        while idx < N and y < 0.95:
            x = 0.05 + (row % 2) * sp / 2.0
            while x < 0.95 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += sp
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        inits.append(c + rng.normal(0, 0.005, c.shape))
        
    # Random feasible starts
    for _ in range(10):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # Phase 1: Multi-start Simulated Annealing
    for c0 in inits:
        c0 = np.clip(c0, 0.02, 0.98)
        c_opt, s_opt = sa_optimize(c0, rng, steps=2500)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: Boundary alignment heuristic
    best_c, best_s = boundary_push(best_c, rng)
    
    # Phase 3: Fine-grained local search with decaying step size
    for scale in [0.002, 0.001, 0.0005]:
        for _ in range(300):
            n_pert = rng.integers(1, 4)
            idxs = rng.choice(N, n_pert, replace=False)
            c_pert = best_c.copy()
            c_pert[idxs] += rng.normal(0, scale, (n_pert, 2))
            c_pert = np.clip(c_pert, 0.005, 0.995)
            s_pert = evaluate_sum_radii(c_pert)
            if s_pert > best_s:
                best_c = c_pert
                best_s = s_pert
                
    # Final radii calculation
    final_r = solve_lp_radii(best_c)
    
    # Strict post-processing to guarantee validator compliance
    for i in range(N):
        ub = min(best_c[i, 0], 1.0 - best_c[i, 0], best_c[i, 1], 1.0 - best_c[i, 1])
        if final_r[i] > ub:
            final_r[i] = max(0.0, ub - 1e-9)
            
    for _ in range(150):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
            if d < final_r[i] + final_r[j] - 1e-12:
                exc = final_r[i] + final_r[j] - d
                final_r[i] -= exc * 0.5
                final_r[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    final_r = np.maximum(final_r, 0.0)
    return best_c, final_r, float(np.sum(final_r))
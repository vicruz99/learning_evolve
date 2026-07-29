import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)
NUM_PAIRS = len(I)

# Precompute LP constraint matrix structure
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J] = 1.0


def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(1e-9, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0


def objective(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])


def constraints(x):
    """Inequality constraints (must be >= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    c_overlap = np.hypot(dx, dy) - (r[I] + r[J])
    
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    return np.concatenate([c_overlap, c_bound])


def force_relax(centers, radii, steps=30):
    """Push overlapping circles apart using repulsive forces."""
    c = centers.copy()
    r = radii.copy()
    n = c.shape[0]
    
    for _ in range(steps):
        moves = np.zeros_like(c)
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-12:
                    overlap = r[i] + r[j] - d
                    shift = overlap / 2.0
                    ux = dx / d
                    uy = dy / d
                    moves[i] += np.array([ux * shift, uy * shift])
                    moves[j] -= np.array([ux * shift, uy * shift])
        
        # Boundary repulsion
        for i in range(n):
            if c[i, 0] < r[i]:
                moves[i, 0] += (r[i] - c[i, 0]) * 0.5
            if c[i, 0] > 1.0 - r[i]:
                moves[i, 0] -= (c[i, 0] - (1.0 - r[i])) * 0.5
            if c[i, 1] < r[i]:
                moves[i, 1] += (r[i] - c[i, 1]) * 0.5
            if c[i, 1] > 1.0 - r[i]:
                moves[i, 1] -= (c[i, 1] - (1.0 - r[i])) * 0.5
        
        c += moves * 0.5
        c = np.clip(c, 1e-6, 1.0 - 1e-6)
    
    return c


def generate_hex(spacing, margin, seed):
    """Generate hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    c = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin
    while idx < N and y < 1.0 - margin:
        x = margin + (row % 2) * spacing / 2
        while x < 1.0 - margin and idx < N:
            c[idx] = [x, y]
            idx += 1
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        c[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    c += rng.normal(0, 0.004, c.shape)
    return np.clip(c, 0.02, 0.98)


def generate_row_pattern(pat, seed):
    """Generate configuration based on row pattern (number of circles per row)."""
    rng = np.random.RandomState(seed)
    c = np.zeros((N, 2))
    idx = 0
    y = 0.05
    dy = 0.88 / (len(pat) - 0.5) if len(pat) > 1 else 0.88
    
    for r_idx, cnt in enumerate(pat):
        shift = 0.0 if r_idx % 2 == 0 else 0.085
        x = 0.05 + shift
        width = 0.9 - 2 * x
        step = width / (cnt - 0.5) if cnt > 1 else 0.0
        
        for _ in range(cnt):
            if idx < N:
                c[idx] = [x, y]
                idx += 1
            x += step
        y += dy
    
    while idx < N:
        c[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    
    c += rng.normal(0, 0.005, c.shape)
    return np.clip(c, 0.02, 0.98)


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    def polish(c0):
        nonlocal best_sum, best_centers, best_radii
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0 * 0.88
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            cx = res.x[0::3]
            cy = res.x[1::3]
            co = np.column_stack((cx, cy))
            ro, so = solve_lp_radii(co)
            if so > best_sum:
                best_sum = so
                best_centers = co.copy()
                best_radii = ro.copy()
        except Exception:
            pass
    
    # Phase 1: Diverse initializations
    
    # 1. Hexagonal lattices with fine parameter sweep
    for sp in np.linspace(0.145, 0.225, 25):
        for margin in np.linspace(0.025, 0.075, 8):
            for seed in range(12):
                c = generate_hex(sp, margin, seed)
                polish(c)
    
    # 2. Structured row patterns
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 5, 4, 4],
        [4, 5, 6, 5, 6], [8, 6, 5, 4, 3], [6, 6, 4, 5, 5],
        [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [5, 5, 6, 5, 5],
        [6, 4, 6, 6, 4], [5, 6, 4, 6, 5], [7, 5, 5, 5, 4],
        [4, 6, 6, 6, 4], [5, 5, 4, 6, 6], [6, 6, 6, 4, 4]
    ]
    for pat in patterns:
        for seed in range(10):
            c = generate_row_pattern(pat, seed)
            polish(c)
    
    # 3. Random configurations
    for seed in range(100):
        rng_s = np.random.RandomState(seed * 17 + 3)
        c = rng_s.uniform(0.05, 0.95, (N, 2))
        polish(c)
    
    # Phase 2: Force-based relaxation + polish
    if best_centers is not None:
        c_relaxed = force_relax(best_centers, best_radii, steps=50)
        polish(c_relaxed)
    
    # Phase 3: Aggressive simulated annealing on centers
    if best_centers is not None:
        current_c = best_centers.copy()
        current_s = best_sum
        
        for step in range(8000):
            temp = 0.02 * np.exp(-step / 1500.0)
            noise = 0.012 * np.sqrt(max(temp, 1e-6))
            
            cp = current_c + rng.normal(0, noise, (N, 2))
            cp = np.clip(cp, 0.01, 0.99)
            
            rp, sp = solve_lp_radii(cp)
            
            delta = sp - current_s
            if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
                current_c = cp
                current_s = sp
                if sp > best_sum:
                    best_sum = sp
                    best_centers = cp.copy()
                    best_radii = rp.copy()
                    polish(cp)
    
    # Phase 4: Individual circle perturbation for fine-tuning
    if best_centers is not None:
        for iteration in range(100):
            improved = False
            for i in range(N):
                for _ in range(8):
                    cp = best_centers.copy()
                    cp[i] += rng.normal(0, 0.004, 2)
                    cp[i] = np.clip(cp[i], 0.01, 0.99)
                    
                    rp, sp = solve_lp_radii(cp)
                    if sp > best_sum:
                        best_sum = sp
                        best_centers = cp.copy()
                        best_radii = rp.copy()
                        improved = True
                        break
                if improved:
                    polish(best_centers)
                    break
    
    # Phase 5: Multi-circle perturbation
    if best_centers is not None:
        for _ in range(200):
            num_pert = rng.choice([1, 2, 3, 4])
            idxs = rng.choice(N, num_pert, replace=False)
            cp = best_centers.copy()
            noise = rng.uniform(0.003, 0.01)
            cp[idxs] += rng.normal(0, noise, (num_pert, 2))
            cp[idxs] = np.clip(cp[idxs], 0.02, 0.98)
            
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_centers = cp.copy()
                best_radii = rp.copy()
                polish(cp)
    
    # Fallback safety net
    if best_centers is None:
        best_centers = generate_hex(0.19, 0.05, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
    
    # Phase 6: Strict post-processing to guarantee validator compliance
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(radii[i], 0.0)
    
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
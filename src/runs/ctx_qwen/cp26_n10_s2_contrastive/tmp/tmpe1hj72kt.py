import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, mx)))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return np.maximum(res.x, 0.0), -res.fun
        except Exception:
            continue
    return np.zeros(n), 0.0


def objective(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])


def constraints(x):
    """Inequality constraints (must be >= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_overlap = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    return np.concatenate([c_overlap, c_bound])


def make_hex_init(spacing, margin, seed):
    """Generate hexagonal lattice initialization with controlled noise."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin + spacing * np.sqrt(3) / 4
    while idx < N and y < 1.0 - margin:
        x = margin + (row % 2) * spacing / 2
        while x < 1.0 - margin and idx < N:
            centers[idx] = [x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)]
            idx += 1
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(margin, 1.0 - margin, 2)
        idx += 1
    return np.clip(centers, 0.01, 0.99)


def make_row_pattern_init(pattern, dy_base, shift_amount, seed):
    """Generate row-based pattern initialization."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    num_rows = len(pattern)
    total_h = 1.0 - 2 * 0.04
    dy = total_h / (num_rows - 0.5)
    
    for r_idx, count in enumerate(pattern):
        y = 0.04 + r_idx * dy
        shift = shift_amount * (r_idx % 2)
        x_start = 0.04 + shift
        x_end = 1.0 - 0.04 - shift
        if count > 1:
            step = (x_end - x_start) / (count - 1)
        else:
            step = 0.0
        for c_idx in range(count):
            if idx < N:
                x = x_start + c_idx * step
                centers[idx] = [x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)]
                idx += 1
    
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    return np.clip(centers, 0.02, 0.98)


def make_corner_edge_init(seed):
    """Generate configuration emphasizing corners and edges."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    
    # Corners
    for x_c in [0.06, 0.94]:
        for y_c in [0.06, 0.94]:
            if idx < N:
                centers[idx] = [x_c, y_c]
                idx += 1
    
    # Edges
    for t in np.linspace(0.15, 0.85, 5):
        if idx < N: centers[idx] = [t, 0.06]; idx += 1
        if idx < N: centers[idx] = [t, 0.94]; idx += 1
    for t in np.linspace(0.15, 0.85, 5):
        if idx < N: centers[idx] = [0.06, t]; idx += 1
        if idx < N: centers[idx] = [0.94, t]; idx += 1
    
    # Interior fill
    while idx < N:
        centers[idx] = rng.uniform(0.2, 0.8, 2)
        idx += 1
    
    centers += rng.normal(0, 0.004, centers.shape)
    return np.clip(centers, 0.03, 0.97)


def make_spiral_init(seed):
    """Generate golden spiral initialization."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    golden_angle = math.pi * (3 - math.sqrt(5))
    for i in range(N):
        r = 0.38 * math.sqrt(i + 1) / math.sqrt(N)
        theta = (i + 1) * golden_angle
        centers[i] = [0.5 + r * math.cos(theta), 0.5 + r * math.sin(theta)]
    centers += rng.normal(0, 0.004, centers.shape)
    return np.clip(centers, 0.03, 0.97)


def try_slqp_polish(c0, r0, rng):
    """Try SLSQP joint optimization from given starting point."""
    noise_scale = rng.uniform(0.0, 0.002)
    c_pert = c0 + rng.normal(0, noise_scale, c0.shape)
    c_pert = np.clip(c_pert, 0.02, 0.98)
    
    r_pert = np.maximum(r0 * 0.93, 1e-5)
    x0 = np.zeros(3 * N)
    x0[0::3] = c_pert[:, 0]
    x0[1::3] = c_pert[:, 1]
    x0[2::3] = r_pert
    
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                       constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
        cx = res.x[0::3]
        cy = res.x[1::3]
        co = np.column_stack((cx, cy))
        ro, so = solve_lp_radii(co)
        return co, ro, so
    except Exception:
        return None, None, 0.0


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    """
    rng = np.random.RandomState(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # === Phase 1: Generate diverse initial configurations ===
    inits = []
    
    # Hexagonal lattices with various spacings and margins
    for sp in np.linspace(0.155, 0.250, 22):
        for margin in [0.025, 0.04, 0.055, 0.07]:
            for seed in range(5):
                inits.append(make_hex_init(sp, margin, seed))
    
    # Row patterns (these worked well in past solutions)
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 5, 4, 4], [4, 5, 6, 5, 6],
        [8, 6, 5, 4, 3], [6, 6, 4, 5, 5], [5, 5, 5, 5, 6], [7, 5, 5, 5, 4],
        [6, 5, 5, 5, 5], [5, 5, 6, 5, 5], [6, 5, 6, 4, 5], [5, 6, 5, 5, 5],
        [4, 6, 6, 5, 5], [5, 4, 6, 6, 5], [6, 6, 5, 5, 4], [7, 4, 6, 5, 4],
        [8, 5, 5, 5, 3], [4, 5, 5, 6, 6], [5, 6, 4, 6, 5]
    ]
    for pat in patterns:
        for dy_base in [0.0, 0.02]:
            for shift in [0.03, 0.05, 0.07, 0.09]:
                for seed in range(6):
                    inits.append(make_row_pattern_init(pat, dy_base, shift, seed))
    
    # Corner-edge patterns
    for seed in range(15):
        inits.append(make_corner_edge_init(seed))
    
    # Spiral patterns
    for seed in range(15):
        inits.append(make_spiral_init(seed))
    
    # Random placements
    for seed in range(50):
        rng_s = np.random.RandomState(seed * 7 + 3)
        inits.append(rng_s.uniform(0.05, 0.95, (N, 2)))
    
    # === Phase 2: Optimize each initialization with SLSQP ===
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0, 1e-4)
        
        co, ro, so = try_slqp_polish(c0, r0, rng)
        if co is not None and so > best_sum:
            best_sum = so
            best_centers = co.copy()
            best_radii = ro.copy()
    
    # === Phase 3: Simulated Annealing Basin Hopping ===
    if best_centers is not None:
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_sum = best_sum
        
        for step in range(800):
            temp = 0.018 * np.exp(-step / 150.0)
            noise = 0.012 * np.exp(-step / 200.0)
            
            cp = current_c + rng.normal(0, noise, (N, 2))
            cp = np.clip(cp, 0.015, 0.985)
            
            rp, sp = solve_lp_radii(cp)
            
            # Simulated annealing acceptance
            if sp > current_sum or rng.random() < np.exp((sp - current_sum) / max(temp, 1e-8)):
                current_c = cp
                current_r = rp
                current_sum = sp
                
                if sp > best_sum:
                    best_sum = sp
                    best_centers = cp.copy()
                    best_radii = rp.copy()
                    
                    # Polish with SLSQP after improvement
                    co_p, ro_p, so_p = try_slqp_polish(cp, rp, rng)
                    if so_p > best_sum:
                        best_sum = so_p
                        best_centers = co_p.copy()
                        best_radii = ro_p.copy()
                        current_c = co_p
                        current_r = ro_p
                        current_sum = so_p
    
    # === Phase 4: Single-circle perturbation ===
    if best_centers is not None:
        for _ in range(500):
            idx = rng.randint(N)
            cp = best_centers.copy()
            cp[idx] += rng.normal(0, 0.008, 2)
            cp = np.clip(cp, 0.02, 0.98)
            
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_centers = cp.copy()
                best_radii = rp.copy()
                
                co_p, ro_p, so_p = try_slqp_polish(cp, rp, rng)
                if so_p > best_sum:
                    best_sum = so_p
                    best_centers = co_p.copy()
                    best_radii = ro_p.copy()
    
    # === Phase 5: Pair perturbation ===
    if best_centers is not None:
        for _ in range(300):
            pair_idx = rng.choice(N, 2, replace=False)
            cp = best_centers.copy()
            cp[pair_idx] += rng.normal(0, 0.006, (2, 2))
            cp = np.clip(cp, 0.02, 0.98)
            
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_centers = cp.copy()
                best_radii = rp.copy()
                
                co_p, ro_p, so_p = try_slqp_polish(cp, rp, rng)
                if so_p > best_sum:
                    best_sum = so_p
                    best_centers = co_p.copy()
                    best_radii = ro_p.copy()
    
    # === Phase 6: Multi-scale fine local search ===
    if best_centers is not None:
        for scale in [0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002]:
            for _ in range(50):
                cp = best_centers + rng.normal(0, scale, (N, 2))
                cp = np.clip(cp, 0.015, 0.985)
                rp, sp = solve_lp_radii(cp)
                if sp > best_sum:
                    best_sum = sp
                    best_centers = cp.copy()
                    best_radii = rp.copy()
                    # Polish
                    co_p, ro_p, so_p = try_slqp_polish(cp, rp, rng)
                    if so_p > best_sum:
                        best_sum = so_p
                        best_centers = co_p.copy()
                        best_radii = ro_p.copy()
    
    # === Phase 7: Triple perturbation ===
    if best_centers is not None:
        for _ in range(200):
            tri_idx = rng.choice(N, 3, replace=False)
            cp = best_centers.copy()
            cp[tri_idx] += rng.normal(0, 0.005, (3, 2))
            cp = np.clip(cp, 0.02, 0.98)
            
            rp, sp = solve_lp_radii(cp)
            if sp > best_sum:
                best_sum = sp
                best_centers = cp.copy()
                best_radii = rp.copy()
    
    # Fallback safety net
    if best_centers is None:
        best_centers = make_hex_init(0.19, 0.05, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
    
    # === Phase 8: Strict post-processing ===
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
                d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-9:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
# sol_000137 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000119 (state ab7c4e6b) state=e231c669 sum of radii=1.903016 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2

# Precompute A_ub structure for LP (constant for fixed N)
A_ub_pairs = np.zeros((NUM_PAIRS, N))
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_ub_pairs[idx, i] = 1.0
        A_ub_pairs[idx, j] = 1.0
        idx += 1

A_ub_bound = np.zeros((4 * N, N))
for i in range(N):
    A_ub_bound[4*i, i] = 1.0
    A_ub_bound[4*i+1, i] = 1.0
    A_ub_bound[4*i+2, i] = 1.0
    A_ub_bound[4*i+3, i] = 1.0

A_ub = np.vstack([A_ub_pairs, A_ub_bound])

def compute_lp_radii_fast(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    c_obj = -np.ones(N)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    b_pairs = dists[np.triu_indices(N, k=1)]
    
    # Boundary distances
    b_bound = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    b_ub = np.concatenate([b_pairs, b_bound])
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*N, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    
    # Fallback to safe radii if LP fails
    safe_r = np.minimum(b_bound, 0.5 * np.min(dists, axis=1)) * 0.95
    return safe_r, np.sum(safe_r)

def obj_centers(c_flat):
    """Objective for center optimization: minimize negative sum of radii from LP."""
    c = c_flat.reshape(N, 2)
    # Ensure centers are strictly inside to avoid degenerate LP bounds
    c = np.clip(c, 1e-6, 1.0 - 1e-6)
    _, s = compute_lp_radii_fast(c)
    return -s

def generate_hex_init(rng, r_est=0.10, pattern=None):
    """Generate a hexagonal lattice initial configuration."""
    if pattern is None:
        pattern = [6, 5, 6, 5, 4]
        
    centers = []
    y = r_est
    for r_idx, cnt in enumerate(pattern):
        shift = r_est if r_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            centers.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3)
        
    c = np.array(centers[:N])
    c += rng.normal(0, 0.005, c.shape)
    c = np.clip(c, 0.02, 0.98)
    return c

def repulsion_init(rng, n_steps=200):
    """Initialize centers using random placement + pairwise repulsion."""
    c = rng.uniform(0.1, 0.9, (N, 2))
    for _ in range(n_steps):
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = c[i] - c[j]
                d = np.linalg.norm(d_vec)
                if d < 0.18 and d > 1e-6:
                    push = (0.18 - d) * 0.5 / d
                    forces[i] += d_vec * push
                    forces[j] -= d_vec * push
        c += forces * 0.01
        c = np.clip(c, 0.02, 0.98)
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    best_centers = None
    best_sum = -1.0
    
    center_bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: Diverse Multi-Start Optimization
    starts = []
    
    # Hexagonal patterns
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4],
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 5, 6],
        [5, 6, 4, 5, 6], [6, 5, 6, 5, 4], [4, 5, 6, 5, 6]
    ]
    for pat in patterns:
        for r_est in [0.09, 0.095, 0.10, 0.105, 0.11]:
            starts.append(generate_hex_init(rng, r_est=r_est, pattern=pat).flatten())
            
    # Repulsion starts
    for _ in range(15):
        starts.append(repulsion_init(rng, n_steps=300).flatten())
        
    # Optimize each start using Powell (derivative-free, robust for non-smooth LP obj)
    for v0 in starts:
        try:
            res = minimize(obj_centers, v0, method='Powell',
                           bounds=center_bounds,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'xtol': 1e-14})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x.reshape(N, 2)
        except Exception:
            continue
            
    if best_centers is None:
        best_centers = starts[0].reshape(N, 2)
        
    # Phase 2: Basin Hopping / Iterative Refinement
    # Perturb and re-optimize to escape local minima
    for step in range(40):
        noise = 0.008 * (0.85 ** step)
        pert_centers = best_centers + rng.normal(0, noise, best_centers.shape)
        pert_centers = np.clip(pert_centers, 0.02, 0.98)
        
        try:
            res = minimize(obj_centers, pert_centers.flatten(), method='Powell',
                           bounds=center_bounds,
                           options={'maxiter': 3000, 'ftol': 1e-14})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x.reshape(N, 2)
        except Exception:
            continue
            
    # Phase 3: LP Refinement & Strict Repair
    radii, final_sum = compute_lp_radii_fast(best_centers)
    
    # Numerical repair to guarantee validation passes within tolerance
    for _ in range(50):
        changed = False
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            x, y = best_centers[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if radii[i] > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return best_centers, radii, final_sum

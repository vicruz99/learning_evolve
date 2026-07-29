# sol_000108 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000076 (state b16097a6) state=de910611 sum of radii=2.577103 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

N = 26

def setup_lp_matrix():
    """Precomputes the sparse-ish structure of the LP inequality matrix."""
    num_pairs = N * (N - 1) // 2
    total_cons = num_pairs + 4 * N
    A_ub = np.zeros((total_cons, N))
    
    # Pair constraints: r_i + r_j <= dist(i,j)
    idx = 0
    pair_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pair_indices.append((i, j))
            idx += 1
            
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(N):
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        
    return A_ub, pair_indices

def solve_lp_and_gradient(centers, A_ub, pair_indices):
    """Solves LP for radii given centers, returns radii, sum, and gradient of sum w.r.t centers."""
    n = centers.shape[0]
    num_pairs = len(pair_indices)
    
    # Precompute distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Construct RHS of inequalities
    b_ub = np.empty(num_pairs + 4 * n)
    idx = 0
    for i, j in pair_indices:
        b_ub[idx] = dists[i, j]
        idx += 1
        
    for i in range(n):
        b_ub[idx] = centers[i, 0]          # r_i <= x
        idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]   # r_i <= 1-x
        idx += 1
        b_ub[idx] = centers[i, 1]          # r_i <= y
        idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]   # r_i <= 1-y
        idx += 1
        
    c_obj = -np.ones(n)  # Maximize sum(r) => Minimize -sum(r)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return None, None, None
            
        radii = res.x
        duals = res.ineqlin.marginals  # Typically <= 0 for <= constraints in minimization
        mu = -duals  # Convert to positive "tightness" forces
        
        grad_centers = np.zeros_like(centers)
        
        # Pairwise gradients: force pushes centers apart proportional to constraint tightness
        idx = 0
        for i, j in pair_indices:
            lam = mu[idx]
            if lam > 1e-8:
                d = dists[i, j]
                if d > 1e-8:
                    vec = centers[i] - centers[j]
                    force = (lam / d) * vec
                    grad_centers[i] += force
                    grad_centers[j] -= force
            idx += 1
            
        # Boundary gradients
        boundary_start = num_pairs
        for i in range(n):
            mu_x = mu[boundary_start + 4*i]
            mu_1x = mu[boundary_start + 4*i + 1]
            mu_y = mu[boundary_start + 4*i + 2]
            mu_1y = mu[boundary_start + 4*i + 3]
            
            # x increases relaxes r<=x, so force +x. 1-x increases relaxes r<=1-x, so force -x.
            grad_centers[i, 0] += mu_x - mu_1x
            grad_centers[i, 1] += mu_y - mu_1y
            
        return radii, np.sum(radii), grad_centers
        
    except Exception:
        return None, None, None

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    
    # 1. Hexagonal lattices with varying densities
    for scale in [0.92, 0.96, 1.0, 1.04]:
        pts = []
        r_est = 0.095
        y = r_est
        row = 0
        while len(pts) < N:
            x = r_est if row % 2 == 0 else 2 * r_est
            while x + r_est <= 1.0 and len(pts) < N:
                pts.append([x * scale, y * scale])
                x += 2 * r_est
            y += r_est * np.sqrt(3)
            row += 1
        inits.append(np.array(pts[:N]))
        
    # 2. Dense random with repulsion initialization
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(50):
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.16:
                        push = (0.16 - dist) * 0.5
                        c[i] += d_vec / dist * push
                        c[j] -= d_vec / dist * push
        inits.append(np.clip(c, 0.05, 0.95))
        
    # 3. Grid-based + perturbation
    grid_x = np.linspace(0.12, 0.88, 5)
    grid_y = np.linspace(0.12, 0.88, 5)
    cx, cy = np.meshgrid(grid_x, grid_y)
    base = np.column_stack((cx.flatten(), cy.flatten()))
    base = np.vstack([base, [0.5, 0.5]])
    for _ in range(5):
        p = base + rng.normal(0, 0.04, base.shape)
        inits.append(np.clip(p, 0.05, 0.95))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    A_ub, pair_indices = setup_lp_matrix()
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    inits = generate_inits(rng)
    
    for init_c in inits:
        centers = init_c.copy()
        velocity = np.zeros_like(centers)
        curr_sum = 0.0
        
        # Gradient ascent phase
        for step in range(1200):
            radii, curr_sum, grad = solve_lp_and_gradient(centers, A_ub, pair_indices)
            if radii is None:
                break
                
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
            # Adaptive step with momentum
            step_size = 0.015 / (1.0 + 0.005 * step)
            velocity = 0.85 * velocity + step_size * grad
            centers += velocity
            centers = np.clip(centers, 0.0, 1.0)
            
            # Periodic random perturbation to escape local minima
            if step % 150 == 0 and step > 0:
                noise = 0.004 * rng.normal(0, 1, centers.shape)
                centers = np.clip(centers + noise, 0.0, 1.0)
                
        # Final LP solve for this trajectory
        radii, curr_sum, _ = solve_lp_and_gradient(centers, A_ub, pair_indices)
        if radii is not None and curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Strict numerical repair to guarantee validation passes
    centers = best_centers
    radii = best_radii
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum

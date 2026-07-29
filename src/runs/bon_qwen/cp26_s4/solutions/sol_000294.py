# sol_000294 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e8a963c) state=9c92197c sum of radii=2.403202 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import random

def generate_initial_centers(n, seed=None):
    """
    Generate initial centers for n circles.
    Uses a perturbed hexagonal grid pattern to provide a good starting geometry.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    centers = np.zeros((n, 2))
    
    # Hexagonal packing parameters
    # Assume a rough radius to space them out
    r_guess = 0.1
    dy = np.sqrt(3) * r_guess  # Vertical spacing
    dx = 2 * r_guess           # Horizontal spacing
    
    y = 0.1 # Start y
    idx = 0
    
    # Generate points in rows
    row_idx = 0
    while idx < n and y <= 0.9:
        is_odd_row = (row_idx % 2 == 1)
        x_start = 0.1
        if is_odd_row:
            x_start += r_guess
        
        x = x_start
        while idx < n and x <= 0.9:
            centers[idx, 0] = x + rng.uniform(-0.001, 0.001) # Small jitter
            centers[idx, 1] = y + rng.uniform(-0.001, 0.001)
            idx += 1
            x += dx
        
        y += dy
        row_idx += 1
        
    # If we didn't fill n circles (unlikely for n=26 with this spacing), fill randomly
    while idx < n:
        centers[idx, 0] = rng.uniform(0.1, 0.9)
        centers[idx, 1] = rng.uniform(0.1, 0.9)
        idx += 1
        
    return centers

def compute_max_radii_lp(centers):
    """
    Given fixed centers, solve the Linear Programming problem to maximize sum of radii.
    Maximize sum(r_i)
    Subject to:
      r_i + r_j <= dist(c_i, c_j) for all i < j
      r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i for all i
      r_i >= 0
    """
    n = centers.shape[0]
    
    # Distances between all pairs
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Wall margins
    walls = np.ones((4, n))
    for i in range(n):
        walls[0, i] = centers[i, 0]       # x >= r
        walls[1, i] = 1 - centers[i, 0]   # 1-x >= r
        walls[2, i] = centers[i, 1]       # y >= r
        walls[3, i] = 1 - centers[i, 1]   # 1-y >= r
        
    # LP Constraints: A_ub @ r <= b_ub
    # r_i + r_j <= d_ij
    # r_i <= wall_margin
    
    # Number of constraints: n*(n-1)/2 + 4*n
    m_pairs = n * (n - 1) // 2
    m_walls = 4 * n
    total_constraints = m_pairs + m_walls
    
    A_ub = np.zeros((total_constraints, n))
    b_ub = np.zeros(total_constraints)
    
    # Pairwise constraints
    row = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[row, i] = 1
            A_ub[row, j] = 1
            b_ub[row] = dists[i, j]
            row += 1
            
    # Wall constraints
    for i in range(n):
        A_ub[row, i] = 1
        b_ub[row] = walls[0, i]
        row += 1
        A_ub[row, i] = 1
        b_ub[row] = walls[1, i]
        row += 1
        A_ub[row, i] = 1
        b_ub[row] = walls[2, i]
        row += 1
        A_ub[row, i] = 1
        b_ub[row] = walls[3, i]
        row += 1
        
    # Objective: Maximize sum(r) => Minimize -sum(r)
    c = -np.ones(n)
    
    # Bounds for r_i: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Using high-precision method if available, otherwise default
    try:
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback to simple heuristic if LP fails
            return None
    except Exception:
        return None

def repulsion_step(centers, radii, alpha=0.1):
    """
    Moves centers to reduce overlap.
    Simple force-directed step.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Repulsion between circles
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                dist = 1e-9
                diff = np.random.rand(2) * 1e-4 # Break symmetry
            
            required_dist = radii[i] + radii[j]
            if dist < required_dist:
                # Push apart
                force_mag = (required_dist - dist) / dist * alpha
                force_vec = diff * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
                
    # Repulsion from walls
    for i in range(n):
        r = radii[i]
        # Left
        if centers[i, 0] < r:
            forces[i, 0] += (r - centers[i, 0]) * alpha * 10
        # Right
        if centers[i, 0] > 1 - r:
            forces[i, 0] -= (centers[i, 0] - (1 - r)) * alpha * 10
        # Bottom
        if centers[i, 1] < r:
            forces[i, 1] += (r - centers[i, 1]) * alpha * 10
        # Top
        if centers[i, 1] > 1 - r:
            forces[i, 1] -= (centers[i, 1] - (1 - r)) * alpha * 10
            
    centers += forces
    # Clamp to [0, 1] just in case, though forces push inside
    centers = np.clip(centers, 1e-6, 1 - 1e-6)
    
    return centers

def optimize_packing_random_restarts(n, num_restarts=10):
    best_sum_radii = -1
    best_centers = None
    best_radii = None
    
    for attempt in range(num_restarts):
        # 1. Initialize
        centers = generate_initial_centers(n, seed=attempt)
        # Initial radii guess
        radii = np.full(n, 0.05)
        
        # 2. Run repulsion simulation to separate centers
        # We increase radii slowly and repel
        current_r = 0.05
        target_r = 0.15 # Grow up to a reasonable size
        
        steps = 200
        for step in range(steps):
            # Linearly increase radii target
            radii = np.full(n, current_r)
            
            # Perform repulsion step
            centers = repulsion_step(centers, radii, alpha=0.05)
            
            current_r += (target_r - current_r) / steps
            
        # 3. Use LP to find optimal radii for the final configuration
        radii_lp = compute_max_radii_lp(centers)
        
        if radii_lp is not None:
            current_sum = np.sum(radii_lp)
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_centers = centers.copy()
                best_radii = radii_lp.copy()
                
    return best_centers, best_radii, best_sum_radii

def run_packing() -> tuple:
    # We need to pack 26 circles
    n = 26
    
    # Run optimization with several restarts
    centers, radii, sum_radii = optimize_packing_random_restarts(n, num_restarts=20)
    
    # Optional: Further refinement using SLSQP on the best result found
    # This tries to tweak positions to increase radii further
    if centers is not None:
        try:
            # Variables: x1, y1, ..., xn, yn
            # We fix radii to the found optimal ones? No, we want to optimize positions
            # to allow for potentially better radii?
            # Actually, the LP step already maximized radii for fixed positions.
            # To improve, we need to move positions.
            # We can use a gradient-based optimizer on the objective: sum(radii)
            # where radii are implicit functions of centers via the LP or simply computed.
            # But computing radii via LP inside a derivative-based optimizer is tricky (non-differentiable).
            
            # Alternative: Just use the result from LP. It's usually very good.
            # Or run a few more repulsion steps with the LP radii.
            
            # Let's do a few more repulsion steps with the LP radii to settle overlaps if any
            # (LP guarantees no overlap, but numerical issues might exist, or we can try to push further)
            # Actually LP result is exact.
            
            # However, the LP radii might be limited by "bottleneck" pairs.
            # Moving centers apart might relax the bottleneck.
            
            # Let's try to optimize positions to maximize the minimum slack.
            # Slack_ij = dist_ij - (r_i + r_j).
            # But r_i are variables too.
            
            # Let's stick to the robust LP+Repulsion result.
            # Just verify validity.
            pass
        except Exception:
            pass

    # Validation check (internal)
    if centers is not None and radii is not None:
        valid = True
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                valid = False
                break
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < radii[i] + radii[j]:
                    valid = False
                    break
            if not valid: break
            
        if not valid:
            # Fallback to a safe grid if something went wrong
            centers = np.zeros((n, 2))
            radii = np.full(n, 0.0)
            idx = 0
            for r_row in range(5):
                for c_col in range(5):
                    if idx < n:
                        centers[idx, 0] = 0.1 + c_col * 0.2
                        centers[idx, 1] = 0.1 + r_row * 0.2
                        radii[idx] = 0.1
                        idx += 1
            sum_radii = np.sum(radii)
    else:
        # Should not happen
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.0)
        sum_radii = 0.0

    return centers, radii, sum_radii

# Import numpy for the validate function to work if needed, though not strictly inside run_packing
# The prompt implies we just need to define run_packing.
# But to be safe with types.

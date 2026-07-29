# sol_000122 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22de7e34) state=147d211f sum of radii=2.623565 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize


def objective_func(vars, n):
    """Objective: maximize sum of radii (minimize negative sum)."""
    radii = vars[2::3]
    return -np.sum(radii)


def constraints_func(vars, n):
    """All inequality constraints as a single array."""
    c = []
    
    # Boundary and non-negativity constraints (5 per circle)
    for i in range(n):
        x = vars[3 * i]
        y = vars[3 * i + 1]
        r = vars[3 * i + 2]
        c.append(x - r)           # x >= r
        c.append(1.0 - x - r)     # 1-x >= r
        c.append(y - r)           # y >= r
        c.append(1.0 - y - r)     # 1-y >= r
        c.append(r)               # r >= 0
    
    # Overlap constraints (one per pair)
    for i in range(n):
        xi = vars[3 * i]
        yi = vars[3 * i + 1]
        ri = vars[3 * i + 2]
        for j in range(i + 1, n):
            dx = xi - vars[3 * j]
            dy = yi - vars[3 * j + 1]
            dist = np.sqrt(dx * dx + dy * dy)
            c.append(dist - ri - vars[3 * j + 2])
    
    return np.array(c)


def make_hexagonal_initial(n):
    """Create hexagonal packing initial configuration."""
    x0 = np.zeros(3 * n)
    
    circles_per_row = [5, 5, 5, 5, 4, 2]
    r_init = 0.075
    
    idx = 0
    for row_idx, count in enumerate(circles_per_row):
        y = (row_idx + 0.5) * (1.0 / 6.0)
        if row_idx % 2 == 0:
            for col in range(count):
                x = (col + 0.5) * (1.0 / count)
                x0[3 * idx] = x
                x0[3 * idx + 1] = y
                x0[3 * idx + 2] = r_init
                idx += 1
        else:
            for col in range(count):
                spacing = 1.0 / (count + 1)
                x = (col + 1) * spacing
                x0[3 * idx] = x
                x0[3 * idx + 1] = y
                x0[3 * idx + 2] = r_init
                idx += 1
    
    return x0


def make_random_initial(n, seed=None):
    """Create random initial configuration with even spacing."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    x0 = np.zeros(3 * n)
    
    cols = 5
    rows = 6
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x0[3 * i] = (col + 0.5) / cols + rng.uniform(-0.03, 0.03)
        x0[3 * i + 1] = (row + 0.5) / rows + rng.uniform(-0.03, 0.03)
        x0[3 * i + 2] = 0.06 + rng.uniform(0, 0.03)
    
    # Clip to valid bounds
    x0[:2 * n] = np.clip(x0[:2 * n], 0.02, 0.98)
    x0[2 * n:] = np.clip(x0[2 * n:], 0.01, 0.3)
    
    return x0


def make_corner_heavy_initial(n):
    """Initial configuration with larger circles in corners."""
    x0 = np.zeros(3 * n)
    
    # Place 4 larger circles in corners
    corner_positions = [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]]
    corner_radii = 0.15
    
    idx = 0
    for pos in corner_positions:
        x0[3 * idx] = pos[0]
        x0[3 * idx + 1] = pos[1]
        x0[3 * idx + 2] = corner_radii
        idx += 1
    
    # Place remaining 22 circles in a grid pattern
    remaining = n - 4
    cols = 5
    rows = 4
    
    for i in range(remaining):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Offset to avoid exact overlap with corners
        if x < 0.4:
            x += 0.1
        if y < 0.4:
            y += 0.05
        x0[3 * idx] = np.clip(x, 0.05, 0.95)
        x0[3 * idx + 1] = np.clip(y, 0.05, 0.95)
        x0[3 * idx + 2] = 0.06
        idx += 1
    
    return x0


def optimize_packing(x0, n, max_iter=5000):
    """Run single optimization from given initial point."""
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))   # x
        bounds.append((0.0, 1.0))   # y
        bounds.append((0.0, 0.5))   # r
    
    result = minimize(
        objective_func,
        x0,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraints_func, 'args': (n,)},
        options={
            'maxiter': max_iter,
            'ftol': 1e-15,
            'disp': False
        }
    )
    return result


def extract_solution(result, n):
    """Extract centers and radii from optimization result."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = result.x[3 * i]
        centers[i, 1] = result.x[3 * i + 1]
        radii[i] = result.x[3 * i + 2]
    return centers, radii


def refine_solution(centers, radii, n, max_iter=3000):
    """Refine a solution from its current state."""
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3 * i] = centers[i, 0]
        x0[3 * i + 1] = centers[i, 1]
        x0[3 * i + 2] = radii[i]
    
    result = optimize_packing(x0, n, max_iter)
    return extract_solution(result, n)


def run_packing():
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -np.inf
    
    # Strategy 1: Hexagonal initial
    for seed in range(5):
        x0 = make_hexagonal_initial(n)
        # Add small perturbation
        x0_perturbed = x0.copy()
        rng = np.random.RandomState(seed * 100 + 42)
        x0_perturbed[:2 * n] += rng.uniform(-0.02, 0.02, 2 * n)
        x0_perturbed[:2 * n] = np.clip(x0_perturbed[:2 * n], 0.01, 0.99)
        x0_perturbed[2 * n:] = np.clip(x0_perturbed[2 * n:], 0.01, 0.4)
        
        result = optimize_packing(x0_perturbed, n, max_iter=6000)
        current_sum = -result.fun
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers, best_radii = extract_solution(result, n)
    
    # Strategy 2: Random initial configurations
    for seed in range(10):
        x0 = make_random_initial(n, seed=seed + 100)
        result = optimize_packing(x0, n, max_iter=6000)
        current_sum = -result.fun
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers, best_radii = extract_solution(result, n)
    
    # Strategy 3: Corner-heavy initial
    for seed in range(3):
        x0 = make_corner_heavy_initial(n)
        rng = np.random.RandomState(seed * 200 + 99)
        x0_perturbed = x0.copy()
        x0_perturbed[:2 * n] += rng.uniform(-0.015, 0.015, 2 * n)
        x0_perturbed[:2 * n] = np.clip(x0_perturbed[:2 * n], 0.01, 0.99)
        x0_perturbed[2 * n:] = np.clip(x0_perturbed[2 * n:], 0.01, 0.4)
        
        result = optimize_packing(x0_perturbed, n, max_iter=6000)
        current_sum = -result.fun
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers, best_radii = extract_solution(result, n)
    
    # Refinement: run optimization again from best solution with higher precision
    if best_centers is not None:
        refined_centers, refined_radii = refine_solution(
            best_centers, best_radii, n, max_iter=8000
        )
        refined_sum = np.sum(refined_radii)
        if refined_sum > best_sum:
            best_centers = refined_centers
            best_radii = refined_radii
            best_sum = refined_sum
    
    # Second refinement pass
    if best_centers is not None:
        refined_centers, refined_radii = refine_solution(
            best_centers, best_radii, n, max_iter=10000
        )
        refined_sum = np.sum(refined_radii)
        if refined_sum > best_sum:
            best_centers = refined_centers
            best_radii = refined_radii
            best_sum = refined_sum
    
    return best_centers, best_radii, best_sum

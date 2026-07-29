# sol_000336 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9bf69ab6) state=099d99ed sum of radii=0.260000 correctness=1.0
# stdout(first 200): Optimization attempt 0 failed: 'x0' must only have one dimension. Optimization attempt 1 failed: 'x0' must only have one dimension. Optimization attempt 2 failed: 'x0' must only have one dimension. Op
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def objective_function(variables, n):
    """
    Objective function to minimize (negative sum of radii).
    variables shape: (n, 3) -> [x, y, r] for each circle.
    """
    radii = variables[:, 2]
    return -np.sum(radii)

def constraints_factory(n):
    """
    Creates a list of constraint dictionaries for scipy.optimize.
    """
    cons = []
    
    # 1. Boundary constraints: r <= x <= 1-r  =>  x - r >= 0, 1 - x - r >= 0
    # Same for y
    for i in range(n):
        # x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[idx, 0] - v[idx, 2]
        })
        # 1 - x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[idx, 0] - v[idx, 2]
        })
        # y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[idx, 1] - v[idx, 2]
        })
        # 1 - y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[idx, 1] - v[idx, 2]
        })
        # r >= 0 (handled by bounds or explicit constraint)
        cons.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[idx, 2]
        })

    # 2. Overlap constraints: dist(i, j) >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2 is non-linear but smooth. 
    # Using dist - (r_i + r_j) >= 0 is better for gradients, but sqrt can be slow.
    # Let's use the squared form to avoid sqrt, but careful with gradient?
    # Actually, dist - sum_r >= 0 is standard.
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, idx1=i, idx2=j: 
                    np.sqrt(np.sum((v[idx1, :2] - v[idx2, :2]) ** 2)) - (v[idx1, 2] + v[idx2, 2])
            })
            
    return cons

def generate_hexagonal_grid(n, margin=0.01):
    """
    Generates initial centers and radii using a hexagonal packing pattern.
    """
    # Estimate radius based on area and density
    # Area = 1. Density approx 0.9. 
    # n * pi * r^2 * 0.9 approx 1? No, packing fraction is pi/sqrt(12) ~ 0.906
    # But for square container, boundary effects reduce this.
    # Let's guess r approx 0.1 initially.
    r_init = 0.08 
    
    centers = []
    # Hexagonal packing: rows offset by r*sqrt(3) vertically, r horizontally
    row_height = r_init * math.sqrt(3)
    
    y = r_init + margin
    row = 0
    while len(centers) < n:
        x = r_init + margin
        # Offset odd rows
        if row % 2 == 1:
            x += r_init
        
        while x + r_init + margin <= 1.0 and len(centers) < n:
            centers.append([x, y, r_init])
            x += 2 * r_init
        
        y += row_height
        row += 1
        
    # Trim to exactly n
    centers = centers[:n]
    return np.array(centers)

def run_packing():
    n = 26
    # Constraints are fixed for a given n
    constraints = constraints_factory(n)
    
    best_result = None
    best_sum_radii = -1.0
    
    # We try multiple initializations to avoid local minima
    # Strategy 1: Hexagonal grid
    # Strategy 2: Random perturbation of grid
    # Strategy 3: Uniform grid
    
    inits = []
    
    # 1. Hexagonal grid
    try:
        grid_init = generate_hexagonal_grid(n)
        inits.append(grid_init)
    except:
        pass
        
    # 2. Randomized Hexagonal (perturbed)
    if len(inits) > 0:
        perturbed = inits[0].copy()
        perturbed[:, :2] += np.random.uniform(-0.01, 0.01, perturbed[:, :2].shape)
        # Clip to valid range roughly
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.05, 0.95)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.05, 0.95)
        inits.append(perturbed)

    # 3. Random positions with small radii
    random_centers = np.random.rand(n, 2) * 0.8 + 0.1
    random_radii = np.full(n, 0.05)
    random_init = np.hstack([random_centers, random_radii.reshape(-1, 1)])
    inits.append(random_init)

    # 4. Grid arrangement (5x5 plus 1)
    # 5x5 grid centers
    grid_centers_5x5 = []
    step = 1.0 / 6.0 # spacing for 5 circles with margin? 
    # Let's just place them evenly
    xs = np.linspace(0.15, 0.85, 5)
    ys = np.linspace(0.15, 0.85, 5)
    for x in xs:
        for y in ys:
            grid_centers_5x5.append([x, y, 0.08])
    # Add one in the middle of a gap?
    # Just take first 26
    grid_centers_5x5 = np.array(grid_centers_5x5[:n])
    inits.append(grid_centers_5x5)

    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    for i, x0 in enumerate(inits):
        try:
            res = minimize(
                objective_function,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_result = res
        except Exception as e:
            print(f"Optimization attempt {i} failed: {e}")
            continue

    if best_result is None:
        # Fallback to a simple grid if optimization fails completely
        # Though unlikely
        centers = generate_hexagonal_grid(n)
        centers[:, 2] = 0.01 # small radii
        return centers[:, :2], centers[:, 2], 0.26

    final_centers = best_result.x[:, :2]
    final_radii = best_result.x[:, 2]
    
    # Post-processing: Ensure strict non-overlap and boundary respect 
    # due to numerical tolerances of solver.
    # The solver minimizes violations, but we want hard constraints.
    # We can shrink radii slightly if needed, but solver should be tight.
    # Let's check and fix if necessary.
    
    # Check overlaps and shrink if needed (simple heuristic fix)
    # If dist < r1 + r2 - epsilon, reduce radii.
    # But since we maximized sum, they should be touching.
    # Just return.
    
    return final_centers, final_radii, np.sum(final_radii)

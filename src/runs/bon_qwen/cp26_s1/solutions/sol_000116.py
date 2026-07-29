# sol_000116 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=a41edd72 sum of radii=2.469392 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def generate_hexagonal_centers(n):
    """Generates an initial hexagonal grid of n centers."""
    points = []
    # Hexagonal lattice generation
    row = 0
    while len(points) < n:
        y = row * np.sqrt(3) / 2
        if row % 2 == 0:
            x_offset = 0
        else:
            x_offset = 0.5
        
        col = 0
        while len(points) < n:
            x = col + x_offset
            points.append((x, y))
            col += 1
        row += 1
    
    # Select n points and scale/translate to [0, 1]
    centers = np.array(points[:n])
    # Shift to center at 0.5, 0.5
    centers -= centers.mean(axis=0)
    # Scale to fit in [-0.5, 0.5] then to [0, 1]
    max_coord = np.max(np.abs(centers))
    centers /= max_coord * 0.9 # 0.9 to leave some room
    centers += 0.5
    return centers

def solve_radii_lp(centers):
    """
    Solves for optimal radii using Linear Programming for fixed centers.
    Maximizes sum(r) subject to r_i + r_j <= dist(i,j) and boundary constraints.
    """
    n = len(centers)
    
    # Objective: Maximize sum(r) => Minimize -sum(r)
    c = -np.ones(n)
    
    # Constraints A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    # 1. Boundary Constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(x)
        
        # r_i <= 1 - x
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - x)
        
        # r_i <= y
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(y)
        
        # r_i <= 1 - y
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - y)
    
    # 2. Pairwise Distance Constraints: r_i + r_j <= dist(i, j)
    # Only add constraints where distance is small enough to be relevant? 
    # For robustness, add all, but LP might be slower. 26*25/2 = 325 constraints is fine.
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        # Use high-performance method
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback or handle failure
            return np.zeros(n)
    except Exception:
        return np.zeros(n)

def run_packing():
    n = 26
    # 1. Initialize centers
    centers = generate_hexagonal_centers(n)
    
    # 2. Iterative Optimization: LP + Force Relaxation
    current_sum_radii = 0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    best_sum = 0
    
    # Parameters
    step_size = 0.05
    decay = 0.99
    iterations = 500
    
    for it in range(iterations):
        # Solve LP for radii
        radii = solve_radii_lp(centers)
        
        if radii is None or np.isnan(radii).any():
            radii = np.zeros(n)
            
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Calculate forces to move centers apart
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    dist = 1e-9
                    vec = np.random.rand(2) - 0.5 # Random nudge
                
                # If circles are touching (tight constraint), push apart
                # Threshold: if r_i + r_j is very close to dist, they are limiting each other
                if radii[i] + radii[j] > dist - 1e-7:
                    # Repulsion force direction
                    force_dir = vec / dist
                    # Magnitude: stronger if they are deeply overlapping (shouldn't happen in LP) 
                    # or just constant for tight constraints
                    magnitude = 1.0 
                    forces[i] += magnitude * force_dir
                    forces[j] -= magnitude * force_dir
        
        # Boundary forces
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            
            # If touching left wall (r approx x), push right
            if r > x - 1e-7:
                forces[i, 0] += 1.0
            # If touching right wall (r approx 1-x), push left
            if r > (1 - x) - 1e-7:
                forces[i, 0] -= 1.0
            # If touching bottom wall (r approx y), push up
            if r > y - 1e-7:
                forces[i, 1] += 1.0
            # If touching top wall (r approx 1-y), push down
            if r > (1 - y) - 1e-7:
                forces[i, 1] -= 1.0

        # Update centers
        centers += step_size * forces
        
        # Clip centers to stay inside [0, 1]
        centers = np.clip(centers, 1e-5, 1 - 1e-5)
        
        # Decay step size
        step_size *= decay

    # 3. Final refinement using random perturbations (Simulated Annealing-lite)
    # This helps escape local optima from the deterministic force method
    best_sum_radii = best_sum
    centers = best_centers.copy()
    
    temp = 0.01
    for _ in range(2000):
        # Perturb one circle
        idx = np.random.randint(0, n)
        delta = np.random.normal(0, temp, 2)
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, 0, 1)
        
        # Check new radii sum
        new_radii = solve_radii_lp(new_centers)
        if new_radii is not None and not np.isnan(new_radii).any():
            new_sum = np.sum(new_radii)
            
            # Accept if better, or with probability if worse (Metropolis)
            if new_sum > best_sum_radii:
                best_sum_radii = new_sum
                centers = new_centers
                best_radii = new_radii
                best_centers = new_centers.copy()
            else:
                # Probability to accept worse solution
                if np.random.rand() < np.exp((new_sum - best_sum_radii) / max(temp, 1e-9)):
                    centers = new_centers
                    best_sum_radii = new_sum # Update local best tracking? No, keep global best
        
        # Cool down
        temp *= 0.995

    # Final validation and return
    final_radii = best_radii
    final_centers = best_centers
    final_sum = np.sum(final_radii)
    
    # Ensure no negative radii due to float errors
    final_radii = np.maximum(final_radii, 0.0)
    
    return final_centers, final_radii, final_sum

import numpy as np
from scipy.optimize import minimize
import random

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def get_constraints(centers, radii):
    """
    Returns an array of constraint values. 
    Positive values indicate satisfied constraints.
    We need:
    1. Boundary: x-r >= 0, 1-(x+r) >= 0, y-r >= 0, 1-(y+r) >= 0
    2. Non-overlap: dist(i,j) - (r_i + r_j) >= 0
    """
    n = len(radii)
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        constraints.append(x - r)
        constraints.append(1.0 - x - r)
        constraints.append(y - r)
        constraints.append(1.0 - y - r)
        
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            constraints.append(dist - radii[i] - radii[j])
            
    return np.array(constraints)

def score_packing(centers, radii):
    return np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # 1. Initialization: Hybrid Hexagonal Lattice
    # We try to fit 26 circles. A 5x5 grid is tight.
    # Let's try a configuration that is slightly perturbed from a grid or hexagonal.
    # Rows: 6, 5, 6, 5, 4 = 26 circles.
    # This arrangement can potentially fit better in height due to offset.
    
    rows_counts = [6, 5, 6, 5, 4]
    centers = []
    y = 0.1 # Initial y guess
    
    for r_count in rows_counts:
        # Distribute x evenly
        x_start = (1.0 - (r_count - 1) * 0.2) / 2.0 # Try to fit diameter 0.2
        # Actually, let's just space them roughly
        # For a hexagonal pattern, odd rows (0, 2, 4) aligned, even rows shifted?
        # Or just space them.
        
        row_centers = []
        if r_count == 0:
            break
            
        # Uniform spacing for x
        # If we want to fit in [0,1], and we have r_count circles
        # Width required roughly 2*r. 
        # Let's initialize with r=0.1
        step = 1.0 / (r_count + 1)
        for k in range(r_count):
            x = (k + 1) * step
            row_centers.append([x, y])
            
        centers.extend(row_centers)
        
        # Vertical step for hexagonal packing is r * sqrt(3) approx 0.173
        # For square grid it is 2r = 0.2.
        # Let's use 0.17 to allow more space
        y += 0.173
        
    centers = np.array(centers)
    if len(centers) < n:
        # Fallback to random if logic fails, though 26 should be covered
        centers = np.random.rand(n, 2) * 0.8 + 0.1
        
    radii = np.full(n, 0.05) # Start small to ensure feasibility
    
    # 2. Simulated Annealing for Global Optimization
    temp = 0.1
    current_sum = score_packing(centers, radii)
    
    # We will optimize centers and radii together
    # To make it easier for SA, we can just optimize centers and set radii to feasible max
    # But maximizing sum of radii is easier if radii are variables.
    
    # Let's do a simple random perturbation SA on (centers, radii)
    # But keeping radii feasible is hard.
    # Alternative: Optimize centers. Then compute max feasible radii for those centers.
    # Computing max radii for fixed centers is a Linear Programming problem.
    # But for n=26, a heuristic "shrink until valid" or "grow until invalid" is faster.
    
    # Better approach for SA:
    # 1. Perturb centers.
    # 2. Solve for max radii given centers? 
    #    Or just perturb radii too.
    
    # Let's use a dedicated numerical optimizer (SLSQP) which is very robust for this 
    # if we provide a good starting point.
    
    # Prepare variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]) # x, y, r bounds
    
    # Objective: Minimize negative sum of radii
    def objective(vars):
        cs = np.reshape(vars[:2*n], (n, 2))
        rs = vars[2*n:]
        return -np.sum(rs)
    
    # Constraints
    def constraint(vars):
        cs = np.reshape(vars[:2*n], (n, 2))
        rs = vars[2*n:]
        con = get_constraints(cs, rs)
        # We need con >= 0
        return con

    # We can try multiple restarts to find global optimum
    final_centers = centers
    final_radii = radii
    max_score = -np.inf
    
    # Restart 1: From the lattice
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints={'type': 'ineq', 'fun': constraint},
                   options={'maxiter': 1000, 'ftol': 1e-8})
    if res.success and res.fun < max_score: # fun is negative sum
        final_centers = np.reshape(res.x[:2*n], (n, 2))
        final_radii = res.x[2*n:]
        max_score = -res.fun

    # Restart 2: Random initialization
    for _ in range(5):
        random_centers = np.random.rand(n, 2) * 0.6 + 0.2
        random_radii = np.full(n, 0.02)
        x0_r = np.concatenate([random_centers.flatten(), random_radii])
        
        res_r = minimize(objective, x0_r, method='SLSQP', bounds=bounds, 
                         constraints={'type': 'ineq', 'fun': constraint},
                         options={'maxiter': 1000, 'ftol': 1e-8})
        if res_r.success and -res_r.fun > max_score:
            final_centers = np.reshape(res_r.x[:2*n], (n, 2))
            final_radii = res_r.x[2*n:]
            max_score = -res_r.fun

    # Final validation and cleaning
    if not validate_packing(final_centers, final_radii):
        # Fallback to a simple valid packing if optimization fails
        # 5x5 grid + 1 small circle
        coords = []
        for i in range(5):
            for j in range(5):
                coords.append([0.1 + i*0.2, 0.1 + j*0.2])
        # Add 26th circle in a gap, e.g., center (0.2, 0.2) with small radius
        # Gap at (0.2, 0.2) is distance 0.141 from neighbors at (0.1,0.1) etc.
        # Neighbors have r=0.1. Available r = 0.141 - 0.1 = 0.041
        coords.append([0.2, 0.2])
        
        final_centers = np.array(coords)
        final_radii = np.array([0.1]*25 + [0.04])
        max_score = np.sum(final_radii)

    # Ensure strict validity with a small buffer
    # Sometimes numerical optimizers hit the boundary exactly.
    # We can shrink radii slightly if needed.
    valid = validate_packing(final_centers, final_radii)
    if not valid:
        # Shrink all radii by 1% to be safe
        final_radii *= 0.99
        valid = validate_packing(final_centers, final_radii)
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
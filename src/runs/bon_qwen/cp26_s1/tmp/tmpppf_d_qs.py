import numpy as np
from scipy.optimize import linprog, differential_evolution
from scipy.spatial.distance import pdist, squareform
import math

def get_hexagonal_centers(n, scale=1.0):
    """
    Generates n centers arranged in a hexagonal lattice pattern.
    """
    # Estimate grid size for hexagonal packing
    # Area per circle in hex packing ~ 2*sqrt(3)*r^2. 
    # But here we just place centers.
    # We want to fit n circles.
    # Approximate number of rows
    rows = math.ceil(math.sqrt(n * 2 / math.sqrt(3)))
    
    centers = []
    count = 0
    r = 0 # placeholder
    
    # We will determine the scale later, but first let's just generate indices
    # Or better, generate points and then scale them to fit in [0,1]
    
    points = []
    for i in range(rows):
        for j in range(rows):
            if count >= n:
                break
            x = j * 1.5 # horizontal spacing factor
            y = i * math.sqrt(3)/2 * 2 # vertical spacing factor? 
            # Standard hex lattice: (i*1.5, j*sqrt(3)) or similar.
            # Let's use:
            # x = j * 1.0
            # y = i * sqrt(3)
            # shift odd rows
            if i % 2 == 1:
                x = j * 1.0 + 0.5
            else:
                x = j * 1.0
            y = i * (math.sqrt(3) / 2) * 2 # Wait, vertical dist is sqrt(3)*r if r=0.5?
            # Let's just use coordinates relative to unit spacing
            # Let unit distance between centers be 1.
            # Horizontal: 1. Vertical: sqrt(3).
            
            # Correction:
            # Row i. If i is even, y = i * sqrt(3)/2. x = j * 1.
            # If i is odd, y = i * sqrt(3)/2. x = j * 1 + 0.5.
            # Wait, standard triangular lattice:
            # basis vectors (1, 0) and (1/2, sqrt(3)/2).
            # points = i*(1,0) + j*(0.5, sqrt(3)/2)? No.
            # points = (j + i/2, i * sqrt(3)/2).
            
            x_coord = j + (0.5 if i % 2 == 1 else 0.0)
            y_coord = i * (math.sqrt(3) / 2)
            
            points.append([x_coord, y_coord])
            count += 1
        if count >= n:
            break
            
    points = np.array(points[:n])
    
    # Scale and center to fit in unit square roughly
    if len(points) > 0:
        # Center at origin
        points -= points.mean(axis=0)
        # Scale to fit in [-0.5, 0.5] roughly
        max_val = np.max(np.abs(points))
        if max_val > 0:
            points /= (max_val * 1.2) # Leave some margin
        # Shift to [0, 1]
        points += 0.5
        
    return points

def solve_lp_radii(centers):
    """
    Given centers, solve LP to maximize sum of radii.
    Variables: r_0, ..., r_{n-1}
    Maximize sum(r_i)
    Subject to:
      r_i >= 0
      r_i <= x_i
      r_i <= 1 - x_i
      r_i <= y_i
      r_i <= 1 - y_i
      r_i + r_j <= dist(i, j)
    """
    n = centers.shape[0]
    
    # Objective: max sum(r) -> min -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints A_ub @ r <= b_ub
    # We need to construct matrix rows
    
    rows_A = []
    rows_b = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x  => -r_i >= -x => r_i <= x. Wait, linprog is A_ub r <= b_ub.
        # So 1*r_i <= x.
        row = np.zeros(n)
        row[i] = 1.0
        rows_A.append(row)
        rows_b.append(x)
        
        # r_i <= 1 - x
        rows_A.append(row.copy())
        rows_b.append(1.0 - x)
        
        # r_i <= y
        rows_A.append(row.copy())
        rows_b.append(y)
        
        # r_i <= 1 - y
        rows_A.append(row.copy())
        rows_b.append(1.0 - y)
        
    # Pairwise constraints r_i + r_j <= dist
    # Distance matrix
    dists = squareform(pdist(centers))
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows_A.append(row)
            rows_b.append(dists[i, j])
            
    A_ub = np.array(rows_A)
    b_ub = np.array(rows_b)
    
    # Bounds for variables: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
        return radii, -res.fun
    else:
        # Fallback to small radii if LP fails (should not happen)
        radii = np.zeros(n)
        return radii, 0.0

def objective_function(center_params):
    """
    Wrapper for optimizer. Reshapes 1D array to 2D centers.
    """
    centers = center_params.reshape(-1, 2)
    radii, total_radius = solve_lp_radii(centers)
    return -total_radius # Minimize negative sum

def run_packing():
    # 1. Initial Guess: Hexagonal Lattice
    n = 26
    centers_init = get_hexagonal_centers(n)
    
    # 2. Solve LP for initial centers to get a baseline
    radii_init, sum_init = solve_lp_radii(centers_init)
    
    # 3. Optimization of centers
    # We want to maximize sum of radii, so minimize -sum.
    # Use differential evolution for global search capabilities, 
    # though with 52 dimensions it might be slow. 
    # Let's try a shorter run or use basinhopping?
    # Actually, differential evolution is robust. 
    # We can limit maxiter.
    
    # Bounds for centers: [0, 1] for each coordinate
    bounds_opt = [(0.0, 1.0) for _ in range(2 * n)]
    
    # To make it faster, we might just run a local optimization from the hex grid
    # Or run DE with small pop size.
    
    # Let's try a few random restarts of a local solver (e.g. Nelder-Mead or similar)
    # But linprog makes the function non-smooth. 
    # Maybe just use the hex grid result as it's likely very good.
    # But let's try to perturb and optimize.
    
    # Strategy: Run DE for a few generations to find better peaks
    try:
        res = differential_evolution(objective_function, bounds_opt, popsize=15, maxiter=50, tol=1e-6, seed=42)
        best_centers = res.x.reshape(n, 2)
    except Exception:
        best_centers = centers_init

    # 4. Final Calculation
    final_radii, final_sum = solve_lp_radii(best_centers)
    
    # Validation check (optional but good for sanity)
    # The LP guarantees constraints are satisfied numerically.
    # However, floating point errors might exist.
    # We can clamp radii slightly if needed, but linprog should be precise.
    
    return best_centers, final_radii, final_sum

# Self-correction/Refinement thought process:
# Differential Evolution on 52 variables with an LP inside might be too slow or unstable 
# if not careful. The LP solver is fast, but 50 * 15 * 52 calls might be heavy.
# However, for 26 circles, it should be acceptable within typical time limits (e.g. 10-60s).
# If time is a concern, we can reduce popsize or maxiter.
# Given the prompt doesn't specify strict time limit, but "run_packing" is called once.
# I'll stick with a reasonable DE setting.

# One potential issue: Linprog might return suboptimal solution if degenerate?
# Unlikely for random centers.
# Also, we should ensure centers are not exactly on boundary if it causes issues, 
# but linprog handles <= 0 correctly.

# Let's refine the hexagonal initialization to be slightly more spread out or rotated?
# Standard hex grid is good.

# Final check on constraints:
# Centers must be in [0,1]. Bounds in DE ensure this.
# Radii non-negative. LP bounds ensure this.
# No overlap. LP constraints ensure this.
# Inside square. LP constraints ensure this.

# Return format: centers (26, 2), radii (26,), sum_radii (float).
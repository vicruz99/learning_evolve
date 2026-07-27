import numpy as np
from scipy.optimize import linprog, minimize

def get_max_sum_radii(centers_flat):
    """
    Computes the maximum sum of radii for a given set of 26 circle centers.
    Uses Linear Programming to solve for optimal radii.
    
    Args:
        centers_flat: np.array of shape (52,) representing 26 centers (x,y)
        
    Returns:
        neg_sum_radii: Negative sum of radii (to be minimized)
    """
    n = 26
    centers = centers_flat.reshape((n, 2))
    
    # Variables for LP: r_1, ..., r_26
    # Objective: Maximize sum(r) <=> Minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Constraints Matrix: A_ub @ r <= b_ub
    # We collect constraints in lists first for efficiency
    A_ub_rows = []
    b_ub_rows = []
    
    # 1. Boundary constraints
    # r_i <= x_i  => 1*r_i <= x_i
    # r_i <= 1-x_i => 1*r_i <= 1-x_i
    # r_i <= y_i  => 1*r_i <= y_i
    # r_i <= 1-y_i => 1*r_i <= 1-y_i
    
    # Construct rows for boundaries
    # We can construct the matrix directly or use a list of tuples. 
    # Given n=26, dense matrix construction is fast enough.
    # 4 boundaries per circle * 26 circles = 104 rows
    
    # x constraints
    for i in range(n):
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(centers[i, 0])
        
        # r_i <= 1 - x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0 - centers[i, 0])
        
        # y constraints
        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(centers[i, 1])
        
        # r_i <= 1 - y_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0 - centers[i, 1])
        
    # 2. Pairwise overlap constraints
    # r_i + r_j <= dist(i, j)
    # Number of pairs = 26 * 25 / 2 = 325
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(dist)
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds for radii: r_i >= 0
    bounds = [(0, None)] * n
    
    try:
        # Solve LP
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            sum_radii = -res.fun # res.fun is min of -sum, so -fun is max sum
            return sum_radii
        else:
            # If LP fails (shouldn't happen with feasible 0 solution), return a large penalty
            # or 0. Returning 0 ensures optimizer avoids this region.
            return 0.0 
    except Exception:
        return 0.0

def run_packing():
    # 1. Initial Guess: Hexagonal Grid
    n = 26
    centers = np.zeros((n, 2))
    
    # Parameters for hex grid
    # We want to fit 26 circles. A 5x5 grid is 25.
    # Let's create a grid with spacing that fits roughly 26 circles.
    # Approx radius 0.1 -> diameter 0.2.
    # Hex spacing horizontal = 2r, vertical = sqrt(3)r.
    # Let's try spacing s = 0.22 (slightly larger than optimal diameter to ensure feasibility initially)
    s = 0.22
    
    count = 0
    row = 0
    y = s / 2 # Start with some offset
    
    while count < n and y + s/2 <= 1.0:
        x_start = s / 2 if row % 2 == 1 else 0 # Shift odd rows
        x = x_start
        
        while x + s/2 <= 1.0 and count < n:
            centers[count] = [x, y]
            count += 1
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
        
    # Flatten centers for optimizer
    x0 = centers.flatten()
    
    # 2. Optimization
    # We want to maximize sum_radii, so we minimize -sum_radii.
    # However, our function get_max_sum_radii returns the positive sum.
    # So we minimize -get_max_sum_radii.
    
    def objective_to_minimize(centers_flat):
        return -get_max_sum_radii(centers_flat)
    
    # Bounds for centers: [0, 1] for all x, y
    bnds = [(0, 1) for _ in range(52)]
    
    # Use Nelder-Mead or Powell. Powell is often better for many variables.
    # Nelder-Mead is robust but can be slow in high dimensions.
    # Let's try Nelder-Mead first as it's standard for this type of non-smooth problem.
    # Or use a hybrid: a few iterations of Nelder-Mead.
    
    # To prevent centers from being too close (causing numerical issues in LP),
    # we might want to add a small regularization or just rely on the fact
    # that sum_radii will be 0 if they overlap too much, driving them apart.
    
    result = minimize(objective_to_minimize, x0, method='Nelder-Mead', 
                     options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    
    best_centers = result.x.reshape((26, 2))
    
    # 3. Final LP Solve to get exact radii for the optimized centers
    # We re-run the logic inside get_max_sum_radii to extract radii
    # Since get_max_sum_radii only returns the sum, we need a helper to return radii too.
    # Let's duplicate the LP logic here briefly or modify the function.
    # For cleanliness, let's write a small helper.
    
    final_radii = solve_radii_for_centers(best_centers)
    final_sum = np.sum(final_radii)
    
    return best_centers, final_radii, final_sum

def solve_radii_for_centers(centers):
    """Solves the LP and returns the radii array."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub_rows = []
    b_ub_rows = []
    
    for i in range(n):
        # Boundaries
        for val in [centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1]]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(val)
            
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(dist)
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    bounds = [(0, None)] * n
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x
    else:
        return np.zeros(n)
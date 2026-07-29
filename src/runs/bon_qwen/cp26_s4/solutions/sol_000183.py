# sol_000183 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 771fbeff) state=ebdbe9d2 sum of radii=2.477096 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
from scipy.spatial.distance import cdist

def compute_max_radii(centers):
    """
    Given a set of centers, solves the LP to maximize sum of radii.
    
    Args:
        centers: np.array of shape (n, 2)
        
    Returns:
        radii: np.array of shape (n)
    """
    n = centers.shape[0]
    
    # Objective: maximize sum(r_i) -> minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Constraints: A_ub @ r <= b_ub
    # 1. Wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # 2. Pairwise constraints: r_i + r_j <= dist(i, j)
    
    # We will build the constraints matrix
    # Number of constraints: 4*n (walls) + n*(n-1)/2 (pairs)
    # However, for scipy linprog, it's better to pass them as lists or arrays.
    # Given n=26, n^2 is small, so dense matrix is fine.
    
    num_pair_constraints = n * (n - 1) // 2
    num_wall_constraints = 4 * n
    total_constraints = num_pair_constraints + num_wall_constraints
    
    A_ub = np.zeros((total_constraints, n))
    b_ub = np.zeros(total_constraints)
    
    # Wall constraints
    # r_i <= x_i  => 1*r_i <= x_i
    # r_i <= 1-x_i => 1*r_i <= 1-x_i
    # r_i <= y_i
    # r_i <= 1-y_i
    
    for i in range(n):
        row_idx = i * 4
        x, y = centers[i]
        
        # r_i <= x
        A_ub[row_idx, i] = 1.0
        b_ub[row_idx] = x
        
        # r_i <= 1-x
        A_ub[row_idx + 1, i] = 1.0
        b_ub[row_idx + 1] = 1.0 - x
        
        # r_i <= y
        A_ub[row_idx + 2, i] = 1.0
        b_ub[row_idx + 2] = y
        
        # r_i <= 1-y
        A_ub[row_idx + 3, i] = 1.0
        b_ub[row_idx + 3] = 1.0 - y
        
    # Pairwise constraints
    # r_i + r_j <= dist(i, j)
    # This corresponds to a row with 1 at col i and 1 at col j
    
    pair_idx = 4 * n
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            A_ub[pair_idx, i] = 1.0
            A_ub[pair_idx, j] = 1.0
            b_ub[pair_idx] = dist
            pair_idx += 1
            
    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # method='highs' is usually good, but 'simplex' or 'interior-point' work too.
    # We need to be careful about infeasibility or numerical issues.
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: return small radii if LP fails (shouldn't happen for valid centers)
            return np.full(n, 0.01)
    except Exception:
        return np.full(n, 0.01)

def objective_func(centers_flat):
    """
    Objective function for optimization.
    Minimizes negative sum of radii.
    """
    n = 26
    centers = centers_flat.reshape((n, 2))
    
    # Clip centers to [0, 1] just in case optimizer goes wild
    # Though bounds should handle it.
    centers = np.clip(centers, 0, 1)
    
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    
    return -sum_radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We want to place 26 points in [0,1]^2.
    # Approximate rows: sqrt(26) approx 5.
    # Let's try to fill rows with hexagonal spacing.
    
    centers_init = []
    
    # Heuristic: 5 rows. 
    # Row 0: 6 pts, Row 1: 5 pts, Row 2: 6 pts, Row 3: 5 pts, Row 4: 4 pts?
    # Total 26.
    # Or simpler: just a regular grid perturbed?
    # Let's try a dense grid and pick points? No, need specific count.
    
    # Let's create a hex lattice and select points inside or closest to center?
    # Better: manually construct rows.
    
    # Let's try 5 rows of 5 points = 25, plus 1 point?
    # Or 6, 5, 5, 5, 5 = 26.
    
    # Hexagonal spacing logic:
    # Vertical spacing dy. Horizontal spacing dx = dy / sin(60) ? No.
    # In hex packing, distance between adjacent points is d.
    # Vertical distance between rows is d * sqrt(3)/2.
    # Horizontal distance within row is d.
    # Offset between rows is d/2.
    
    # Let's estimate d.
    # 26 circles. Area approx 1. 26 * pi * (d/2)^2 approx 1?
    # d^2 approx 4 / (26 pi) approx 0.049. d approx 0.22.
    # dy = 0.22 * 0.866 approx 0.19.
    # 5 rows need 4 * dy approx 0.76 vertical space. Fits.
    # 6 points in row need 5 * dx approx 1.1 horizontal space. Too wide.
    # So maybe 5 points in row? 5 points need 4 * dx approx 0.88. Fits.
    # So maybe 5 rows of 5 points? That's 25.
    # Where to put 26th?
    # Maybe 6, 5, 5, 5, 5 is tight horizontally.
    # Maybe 5, 5, 5, 5, 6?
    
    # Let's just use a generic grid initialization which is robust.
    # 6 columns, 5 rows = 30 slots.
    # We pick 26 slots. Which ones?
    # To maximize radii, we want points spread out.
    # Removing points from corners might help?
    # Actually, corners allow large circles?
    # A circle in corner is limited by 2 walls.
    # A circle in center is limited by 0 walls (but neighbors).
    # Usually, filling the square uniformly is good.
    
    # Let's generate a 6x5 grid and drop 4 points randomly or from corners?
    # Dropping from corners might reduce boundary constraints?
    # Actually, keeping corners is good for density.
    # Maybe drop points from the middle? No, that creates large gaps.
    # Maybe just take the first 26 in row-major order?
    # That would leave the last row incomplete.
    
    # Let's try a random initialization + optimization?
    # But deterministic is better for reproducibility.
    
    # Let's create a hexagonal layout.
    # Rows 0..4.
    # Row 0 (even): x = 0, d, 2d, 3d, 4d, 5d ?
    # Let's just place points on a lattice scaled to fit [0,1].
    
    # Generate all lattice points in a slightly larger box, scale to fit?
    # Lattice points: (i*dx + j*dx/2, j*dy)
    # We want to cover [0,1]x[0,1].
    
    # Let's just use a simple grid for initialization.
    # 26 points.
    # Maybe 5 rows: 6, 5, 5, 5, 5 points.
    
    centers_list = []
    
    # Row 0: 6 points
    # x coords: spread in [0,1]
    # y coord: 1/10 (some margin)
    # Actually, let's put them in the middle.
    
    # Let's define a function to get row points
    def get_row_points(num_points, y):
        pts = []
        # Spread uniformly
        for k in range(num_points):
            x = (k + 0.5) / num_points
            pts.append([x, y])
        return pts

    # Let's try to fit 26 points in 5 rows with hex offset
    # Row 0: 6 pts, y=0.1
    # Row 1: 5 pts, y=0.3, x offset by 0.1?
    # Row 2: 6 pts, y=0.5
    # Row 3: 5 pts, y=0.7, x offset
    # Row 4: 4 pts, y=0.9
    
    # Wait, 6+5+6+5+4 = 26.
    # Let's try this.
    
    rows_config = [6, 5, 6, 5, 4]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for r_idx, count in enumerate(rows_config):
        y = ys[r_idx]
        # If odd row index, shift x
        # But uniform spacing for each row is easier for init
        xs = np.linspace(0.5/count, 1 - 0.5/count, count)
        if r_idx % 2 == 1:
            # Shift slightly?
            # Hex packing shift is half spacing.
            # Spacing is 1/(count-1)? No, 1/count roughly.
            # Let's just not shift for init, optimizer will fix it.
            pass
        for x in xs:
            centers_list.append([x, y])
            
    centers_init = np.array(centers_list)
    
    # Ensure we have 26 points
    assert centers_init.shape[0] == n
    
    # 2. Optimization
    # We use Nelder-Mead to optimize centers.
    # Bounds for centers: [0, 1] for x and y.
    bounds_opt = [(0, 1) for _ in range(2 * n)]
    
    # Initial point
    x0 = centers_init.flatten()
    
    # Run optimizer
    # Nelder-Mead doesn't use bounds directly in standard scipy optimize.minimize 
    # without constraints, but we can clamp or use L-BFGS-B?
    # L-BFGS-B needs gradients.
    # Powell is good.
    # Let's try 'Powell' or 'Nelder-Mead'.
    # Since bounds are important, maybe just clip inside objective?
    # Or use 'SLSQP' with constraints? No gradient.
    # Let's use 'Nelder-Mead' and clip centers in objective.
    
    # To make it more robust, let's run a few random restarts or just one good run.
    # Given the function evaluation cost (LP), we should be careful.
    # Nelder-Mead is derivative-free.
    
    res_opt = opt.minimize(objective_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-9})
    
    best_centers = res_opt.x.reshape((n, 2))
    best_centers = np.clip(best_centers, 0, 1) # Safety
    
    # 3. Final Radii Calculation
    final_radii = compute_max_radii(best_centers)
    total_sum = np.sum(final_radii)
    
    return best_centers, final_radii, total_sum

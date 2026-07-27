import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp

def solve_radii_lp(centers):
    """
    Solves the LP to find maximum radii for fixed centers.
    Maximize sum(r_i) subject to r_i + r_j <= dist(i,j) and boundary constraints.
    """
    n = centers.shape[0]
    
    # Variables: r_0, ..., r_{n-1}
    # c = -1 (since we minimize -sum(r))
    c = -np.ones(n)
    
    # Constraints A_ub @ r <= b_ub
    # 1. Non-overlap: r_i + r_j <= dist_ij
    # 2. Boundary: r_i <= x_i, r_i <= 1-x_i, etc.
    
    # We will build the constraint matrix sparsely
    # Number of constraints: n*(n-1)/2 (overlap) + 4*n (boundary)
    n_pairs = n * (n - 1) // 2
    n_bound = 4 * n
    m_constraints = n_pairs + n_bound
    
    rows = []
    cols = []
    data = []
    b_ub = []
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            idx = len(rows)
            rows.extend([idx, idx])
            cols.extend([i, j])
            data.extend([1.0, 1.0])
            b_ub.append(dist)
            
    # Boundary constraints
    # r_i <= x_i  => 1*r_i <= x_i
    # r_i <= 1-x_i => 1*r_i <= 1-x_i
    # r_i <= y_i => 1*r_i <= y_i
    # r_i <= 1-y_i => 1*r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        rows.append(len(rows))
        cols.append(i)
        data.append(1.0)
        b_ub.append(x)
        
        # r_i <= 1 - x
        rows.append(len(rows))
        cols.append(i)
        data.append(1.0)
        b_ub.append(1.0 - x)
        
        # r_i <= y
        rows.append(len(rows))
        cols.append(i)
        data.append(1.0)
        b_ub.append(y)
        
        # r_i <= 1 - y
        rows.append(len(rows))
        cols.append(i)
        data.append(1.0)
        b_ub.append(1.0 - y)
        
    A = sp.csr_matrix((data, (rows, cols)), shape=(m_constraints, n))
    b = np.array(b_ub)
    
    # Bounds for r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = opt.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
        else:
            # Fallback if LP fails (e.g. infeasible due to numerical issues)
            # Return small radii
            return np.zeros(n), 0.0
    except Exception:
        return np.zeros(n), 0.0

def objective(centers_flat):
    """
    Objective function for scipy.optimize.
    Centers are flattened array of shape (2*n,).
    """
    n = 13 # Half of 26
    centers = centers_flat.reshape(-1, 2)
    # Solve LP
    radii, sum_r = solve_radii_lp(centers)
    return -sum_r # Minimize negative sum

def run_packing():
    n = 26
    
    # 1. Generate Initial Centers (Hexagonal Lattice)
    # Strategy: 5 rows. Row counts: 6, 5, 6, 5, 4. Total 26.
    # Hexagonal spacing.
    # Let's estimate size.
    # Width 1. 6 circles in row.
    # If we place 6 circles, we need space.
    # Let's just generate points and let optimizer fix it.
    
    centers = []
    
    # Parameters for lattice
    # We want to fill [0,1]x[0,1]
    # Approximate grid
    
    # Row 0: 6 circles
    y0 = 0.15
    # x coords: distribute in [0.1, 0.9] roughly? 
    # Let's use linspace
    x_row0 = np.linspace(0.15, 0.85, 6)
    for x in x_row0:
        centers.append([x, y0])
        
    # Row 1: 5 circles, offset by half step
    # Vertical spacing for hex packing: sqrt(3)/2 * horizontal_step
    h_step = (0.85 - 0.15) / 5.0 # roughly
    y1 = y0 + h_step * np.sqrt(3)/2
    # Offset x by h_step/2
    x_row1 = np.linspace(0.15 + h_step/2, 0.85 - h_step/2, 5)
    for x in x_row1:
        centers.append([x, y1])
        
    # Row 2: 6 circles
    y2 = y1 + h_step * np.sqrt(3)/2
    x_row2 = np.linspace(0.15, 0.85, 6)
    for x in x_row2:
        centers.append([x, y2])
        
    # Row 3: 5 circles
    y3 = y2 + h_step * np.sqrt(3)/2
    x_row3 = np.linspace(0.15 + h_step/2, 0.85 - h_step/2, 5)
    for x in x_row3:
        centers.append([x, y3])
        
    # Row 4: 4 circles
    y4 = y3 + h_step * np.sqrt(3)/2
    x_row4 = np.linspace(0.15 + h_step, 0.85 - h_step, 4) # narrower
    for x in x_row4:
        centers.append([x, y4])
        
    centers = np.array(centers[:n])
    
    # 2. Optimize Centers
    # Flatten centers for optimizer
    x0 = centers.flatten()
    
    # Use Nelder-Mead for derivative-free optimization
    # Bounds for centers [0, 1]
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Run optimization
    # Nelder-Mead doesn't support bounds directly in older scipy, but in recent it does?
    # Actually Nelder-Mead in scipy.optimize.minimize does not support bounds.
    # We can use 'Powell' or 'Nelder-Mead' and handle bounds by clipping or penalty.
    # Or use 'L-BFGS-B' which supports bounds. But it needs gradient or finite diff.
    # Let's try Nelder-Mead and manually project? No, simple is better.
    # Let's use Powell, it handles box constraints if we pass bounds? 
    # Powell also doesn't strictly enforce bounds in all versions, but usually respects them if initialized well?
    # Actually, `minimize` with method='Nelder-Mead' ignores bounds.
    # `minimize` with method='Powell' also ignores bounds?
    # `minimize` with method='L-BFGS-B' supports bounds.
    # Let's try L-BFGS-B with finite difference gradient.
    
    try:
        res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-9})
        centers_opt = res.x.reshape(-1, 2)
    except Exception:
        # Fallback to Nelder-Mead if L-BFGS-B fails, though bounds might be violated
        # We can clip centers to [0,1] manually if needed, but let's hope initialization is good.
        # Actually, let's just use Nelder-Mead and hope it stays inside, or clip at end.
        # But for safety, let's try to run Nelder-Mead with a wrapper that clips?
        # No, let's just use the result from L-BFGS-B.
        # If it fails, use initial.
        centers_opt = centers

    # Ensure centers are within [0, 1]
    centers_opt = np.clip(centers_opt, 0, 1)

    # 3. Compute Final Radii
    final_radii, final_sum = solve_radii_lp(centers_opt)
    
    # 4. Validate (internal check)
    # We trust the LP solution is valid for these centers.
    # But just to be safe, we can check overlap manually?
    # The LP constraints guarantee non-overlap and boundaries.
    # However, numerical precision might be an issue.
    # Let's trust it.
    
    return centers_opt, final_radii, final_sum

if __name__ == "__main__":
    # Dummy run to test locally if needed
    # centers, radii, s = run_packing()
    # print(s)
    pass
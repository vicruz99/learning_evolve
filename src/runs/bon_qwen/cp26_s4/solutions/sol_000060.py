# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b8e92087) state=250e6b16 sum of radii=2.392000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_radii(centers):
    """Solve LP to find optimal radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)
    
    A_rows = []
    b_vals = []
    
    # Boundary constraints: r_i <= distance to nearest boundary
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0],
                 centers[i, 1], 1.0 - centers[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(mx)
    
    # Non-overlap constraints: r_i + r_j <= distance between centers
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(d)
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    bounds = [(0.0, None)] * n
    
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return -res.fun, res.x
    else:
        r = np.full(n, 0.005)
        return np.sum(r), r

def hex_centers(n, r):
    """Generate hexagonal packing centers with given base radius."""
    pts = []
    row = 0
    while len(pts) < n:
        y = r + row * r * np.sqrt(3)
        if y > 1.0 - r:
            break
        xs = r if row % 2 == 0 else 2 * r
        while xs < 1.0 - r and len(pts) < n:
            pts.append([xs, y])
            xs += 2 * r
        row += 1
    return np.array(pts[:n])

def grid_centers(n, nx):
    """Generate grid layout centers."""
    ny = (n + nx - 1) // ny
    ny = (n + nx - 1) // nx
    xs = np.linspace(1.0 / (2 * nx), 1.0 - 1.0 / (2 * nx), nx)
    ys = np.linspace(1.0 / (2 * ny), 1.0 - 1.0 / (2 * ny), ny)
    pts = []
    for y in ys:
        for x in xs:
            pts.append([x, y])
            if len(pts) == n:
                break
        if len(pts) == n:
            break
    return np.array(pts[:n])

def run_packing():
    n = 26
    best_s = 0.0
    best_c = None
    best_r = None
    
    # Generate candidate center configurations
    cands = []
    
    # Hexagonal layouts with varying densities
    for r in np.arange(0.06, 0.12, 0.004):
        try:
            c = hex_centers(n, r)
            if len(c) == n:
                cands.append(c)
        except:
            pass
    
    # Grid layouts with different aspect ratios
    for nx in range(3, 8):
        try:
            c = grid_centers(n, nx)
            if len(c) == n:
                cands.append(c)
        except:
            pass
    
    # Evaluate all candidates and find best
    for c in cands:
        s, r = solve_radii(c)
        if s > best_s:
            best_s = s
            best_c = c.copy()
            best_r = r.copy()
    
    # Local refinement via random perturbations
    np.random.seed(123)
    for iteration in range(300):
        noise = np.random.randn(n, 2) * 0.04
        perturbed = best_c + noise
        perturbed = np.clip(perturbed, 0.03, 0.97)
        
        s, r = solve_radii(perturbed)
        if s > best_s:
            best_s = s
            best_c = perturbed.copy()
            best_r = r.copy()
    
    return best_c, best_r, best_s

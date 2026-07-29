# sol_000102 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 80fa60f2) state=c108a32d sum of radii=2.626003 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def check_validity(centers, radii):
    """Helper to check validity without printing, for internal use."""
    n = centers.shape[0]
    # Check boundaries
    if np.any(radii < -1e-12):
        return False
    if np.any(centers[:, 0] - radii < -1e-12) or np.any(centers[:, 0] + radii > 1 + 1e-12):
        return False
    if np.any(centers[:, 1] - radii < -1e-12) or np.any(centers[:, 1] + radii > 1 + 1e-12):
        return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9: # slightly stricter for internal check
                return False
    return True

def objective(vars, n=26):
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # We want to maximize sum(r), so minimize -sum(r)
    radii = vars[2::3]
    return -np.sum(radii)

def bounds_def(n=26):
    bounds = []
    for _ in range(n):
        # x in [0, 1]
        bounds.append((0, 1))
        # y in [0, 1]
        bounds.append((0, 1))
        # r in [0, 0.5] (upper bound loose)
        bounds.append((0, 0.5))
    return bounds

def constraints_def(vars, n=26):
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = vars[3*i]
        centers[i, 1] = vars[3*i + 1]
        radii[i] = vars[3*i + 2]
    
    cons = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # Also r >= 0 is handled by bounds
    
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        # x >= r
        cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx] - v[3*idx+2]})
        # x <= 1 - r => x + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1 - v[3*idx] - v[3*idx+2]})
        # y >= r
        cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx+1] - v[3*idx+2]})
        # y <= 1 - r => y + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1 - v[3*idx+1] - v[3*idx+2]})

    # Overlap constraints: dist >= r_i + r_j => dist^2 >= (r_i + r_j)^2
    # To avoid sqrt in constraint function for speed/numerics? 
    # Actually sqrt is fine, or just dist - (r_i+r_j) >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def dist_con(v, i=i, j=j):
                x1, y1, r1 = v[3*i], v[3*i+1], v[3*i+2]
                x2, y2, r2 = v[3*j], v[3*j+1], v[3*j+2]
                d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                return d - (r1 + r2)
            cons.append({'type': 'ineq', 'fun': dist_con})
            
    return cons

def run_packing():
    n = 26
    
    # Generate initial guesses
    # Strategy 1: Grid 5x5 + 1 in center hole
    init1 = np.zeros(3 * n)
    r_start = 0.099
    idx = 0
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            init1[3*idx] = x
            init1[3*idx+1] = y
            init1[3*idx+2] = r_start
            idx += 1
    # 26th circle in a hole, e.g., (0.3, 0.3) is occupied. Hole at (0.2, 0.2)
    # Distance to (0.1, 0.1) is sqrt(2)*0.1 approx 0.141. 
    # If big circles r=0.1, gap is 0.041. Let's place small circle.
    init1[3*25] = 0.2
    init1[3*25+1] = 0.2
    init1[3*25+2] = 0.04

    # Strategy 2: Hexagonal packing approximation
    init2 = np.zeros(3 * n)
    r_hex = 0.09
    idx = 0
    # Try to pack in hexagonal lattice
    # Rows
    row_configs = [
        (5, 0),      # 5 circles, offset 0
        (5, 0.5),    # 5 circles, offset 0.5 (in units of 2r)
        (5, 0),
        (5, 0.5),
        (4, 0),
        (2, 0.5)
    ]
    # This is a rough sketch, let's just generate points in a hex grid and clip
    # Better: Just use a dense random initialization or the grid one is usually sufficient for local opt.
    # Let's stick to optimizing init1.
    
    # We will run optimization on init1.
    # To improve chances, we can scale up radii slightly and let solver fix overlaps?
    # Or just let solver find max sum.
    
    cons = constraints_def(init1, n)
    bnds = bounds_def(n)
    
    # We might want to normalize the problem or ensure good scaling.
    # SLSQP is generally robust.
    
    best_sum = -np.inf
    best_sol = None
    
    # Run optimization
    # Use multiple restarts with slight perturbations
    for trial in range(3):
        x0 = init1.copy()
        if trial > 0:
            # Add noise to centers
            noise = np.random.uniform(-0.01, 0.01, size=(n, 2))
            for i in range(n):
                x0[3*i] += noise[i, 0]
                x0[3*i+1] += noise[i, 1]
                # Ensure inside [0,1]
                x0[3*i] = np.clip(x0[3*i], 0.01, 0.99)
                x0[3*i+1] = np.clip(x0[3*i+1], 0.01, 0.99)
        
        try:
            res = opt.minimize(objective, x0, args=(n,), method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success or res.fun < best_sum: # lower fun is better (max sum)
                # Validate solution manually to be safe
                centers = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
                radii = np.array([res.x[3*i+2] for i in range(n)])
                
                # Fix negative radii due to numerical error
                radii = np.maximum(radii, 1e-9)
                
                # Check if valid
                if check_validity(centers, radii):
                    current_sum = np.sum(radii)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_sol = (centers, radii)
        except Exception as e:
            continue

    if best_sol is None:
        # Fallback to a valid static solution (Grid 5x5 + 1 small)
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        r = 0.1
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx, 0] = 0.1 + i * 0.2
                centers[idx, 1] = 0.1 + j * 0.2
                radii[idx] = r
                idx += 1
        centers[idx, 0] = 0.3 # Place in a gap?
        centers[idx, 1] = 0.3
        radii[idx] = 0.0 # Invalid, need positive.
        # Just return the 5x5 grid with r=0.1 for 25 circles and one tiny one?
        # But we need 26 circles.
        # Let's just place the last one with r=0.001
        radii[idx] = 0.001
        centers[idx] = [0.5, 0.5] # Center
        # Check overlap with center circle (0.5, 0.5) r=0.1? Dist 0.
        # Move it to (0.2, 0.2)
        centers[idx] = [0.2, 0.2]
        # Dist to (0.1, 0.1) is 0.141, r_big=0.1, r_small=0.041.
        radii[idx] = 0.04
        
        best_sum = np.sum(radii)
        best_sol = (centers, radii)

    return best_sol[0], best_sol[1], best_sum

# Ensure no closures or lambdas in global scope if that's an issue, 
# but the prompt says "Make all helper functions top level and have no closures".
# My constraints_def creates lambdas inside. Let's refactor to avoid lambdas.

def constraint_boundary_x_low(v, idx):
    return v[3*idx] - v[3*idx+2]

def constraint_boundary_x_high(v, idx):
    return 1 - v[3*idx] - v[3*idx+2]

def constraint_boundary_y_low(v, idx):
    return v[3*idx+1] - v[3*idx+2]

def constraint_boundary_y_high(v, idx):
    return 1 - v[3*idx+1] - v[3*idx+2]

def constraint_overlap(v, i, j):
    x1, y1, r1 = v[3*i], v[3*i+1], v[3*i+2]
    x2, y2, r2 = v[3*j], v[3*j+1], v[3*j+2]
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return d - (r1 + r2)

def run_packing_refactored():
    n = 26
    init = np.zeros(3 * n)
    r_start = 0.099
    idx = 0
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            init[3*idx] = x
            init[3*idx+1] = y
            init[3*idx+2] = r_start
            idx += 1
    # 26th circle
    init[3*25] = 0.2
    init[3*25+1] = 0.2
    init[3*25+2] = 0.04

    bnds = []
    for _ in range(n):
        bnds.append((0, 1))
        bnds.append((0, 1))
        bnds.append((0, 0.5))

    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_x_low(v, i)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_x_high(v, i)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_y_low(v, i)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_y_high(v, i)})
    
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: constraint_overlap(v, i, j)})

    best_sum = -np.inf
    best_centers = None
    best_radii = None

    # Run a few times
    for trial in range(5):
        x0 = init.copy()
        if trial > 0:
            noise = np.random.uniform(-0.005, 0.005, size=(n, 2))
            for i in range(n):
                x0[3*i] = np.clip(x0[3*i] + noise[i, 0], 0.01, 0.99)
                x0[3*i+1] = np.clip(x0[3*i+1] + noise[i, 1], 0.01, 0.99)
        
        try:
            res = opt.minimize(objective, x0, args=(n,), method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 2000})
            
            centers_cand = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
            radii_cand = np.array([res.x[3*i+2] for i in range(n)])
            radii_cand = np.maximum(radii_cand, 1e-9)
            
            if check_validity(centers_cand, radii_cand):
                s = np.sum(radii_cand)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers_cand
                    best_radii = radii_cand
        except:
            pass

    if best_centers is None:
        # Fallback
        best_centers = init[:2*n].reshape(n, 2)
        best_radii = init[2::3]
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum

# Map run_packing to the refactored one
run_packing = run_packing_refactored

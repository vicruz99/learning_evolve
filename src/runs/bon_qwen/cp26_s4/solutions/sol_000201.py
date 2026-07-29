# sol_000201 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7f4d5c4f) state=fa03d1a9 sum of radii=0.260000 correctness=1.0
# stdout(first 200): Optimization failed: cannot reshape array of size 26 into shape (26,2) Optimization failed: cannot reshape array of size 26 into shape (26,2) Optimization failed: cannot reshape array of size 26 into 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import itertools
import random

def get_distance_matrix(centers):
    """Compute pairwise distances between centers."""
    n = centers.shape[0]
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
    return dists

def objective(params, n):
    """
    Objective function to minimize: -sum(radii)
    params: flattened array [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    """
    centers = params[0::3].reshape(n, 2)
    radii = params[1::3] # Wait, structure needs to be consistent.
    # Let's use structure: [x1, y1, r1, x2, y2, r2, ...] is length 3*n.
    # Actually, let's use params[:2*n] as centers, params[2*n:] as radii.
    # Or reshape.
    # Let's stick to a specific layout in run_packing.
    # Here we just return -sum(radii).
    # The caller must ensure params structure.
    # Let's assume params is [x1, y1, ..., xn, yn, r1, ..., rn]
    # Length 2n + n = 3n.
    radii = params[2*n:]
    return -np.sum(radii)

def constraints_overlap(params, n):
    """
    Constraints: dist(i, j) >= r_i + r_j
    Returns array of constraint values >= 0
    """
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            c = d - (radii[i] + radii[j])
            constraints.append(c)
    return np.array(constraints)

def constraints_boundary(params, n):
    """
    Constraints:
    x - r >= 0
    x + r <= 1  => 1 - x - r >= 0
    y - r >= 0
    y + r <= 1  => 1 - y - r >= 0
    r >= 0
    """
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    constraints = []
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left wall
        constraints.append(x - r)
        # Right wall
        constraints.append(1 - x - r)
        # Bottom wall
        constraints.append(y - r)
        # Top wall
        constraints.append(1 - y - r)
        # Non-negative radius
        constraints.append(r)
        
    return np.array(constraints)

def create_initial_config(n, method='grid'):
    """
    Create initial configuration for n circles.
    Returns array of shape (3n,) -> [x1, y1, ..., xn, yn, r1, ..., rn]
    """
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.01 # Small initial radius
    
    if method == 'grid':
        # Hexagonal grid approximation
        # Estimate radius for equal packing
        # Area ~ 1, density ~ 0.9 -> N * pi * r^2 ~ 0.9 -> r ~ sqrt(0.9 / (N*pi))
        r_est = np.sqrt(0.9 / (n * np.pi))
        # Scale to fit in square roughly
        # For grid, spacing ~ 2r.
        # Let's just place them in a grid and let optimizer adjust.
        
        # Try to fit sqrt(N) x sqrt(N) grid
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Spacing
        dx = 1.0 / (cols + 1)
        dy = 1.0 / (rows + 1)
        
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < n:
                    centers[idx] = [dx * (c + 1), dy * (r + 1)]
                    idx += 1
    elif method == 'hex':
        # Hexagonal packing initialization
        # Rows with alternating shifts
        # Estimate row count
        # Area of hex cell ~ 2.6 r^2. N * 2.6 r^2 ~ 1 -> r ~ 0.6 / sqrt(N)
        r_est = 0.6 / np.sqrt(n)
        
        # Vertical spacing r*sqrt(3)
        # Horizontal spacing 2r
        # Let's try to determine number of rows
        # Approx height 1. rows * r*sqrt(3) ~ 1 -> rows ~ 1/(r*1.732) ~ 0.57/r
        # Approx width 1. cols * 2r ~ 1 -> cols ~ 1/(2r) ~ 0.5/r
        
        # Just generate a pattern
        count = 0
        row_idx = 0
        while count < n:
            # Determine number of circles in this row
            # Alternate between k and k-1 or similar
            # Let's just fit as many as possible in width 1
            # Max cols = floor(1 / (2*r_est)) + 1 ?
            # Let's just use a fixed pattern for 26
            # 5, 4, 5, 4, 5, 3 ?
            # Or just fill row by row
            
            # Simple row filling
            # Spacing
            spacing_x = 2 * r_est
            spacing_y = r_est * np.sqrt(3)
            
            # Shift for odd rows
            shift = spacing_x / 2 if row_idx % 2 == 1 else 0
            
            # Max circles in row
            # width available ~ 1
            # centers from shift to 1-shift?
            # Let's just place centers
            # Start x
            x_start = shift + r_est # Ensure inside? No, optimizer handles boundaries.
            # Actually just place them.
            
            # Let's define row positions
            y_pos = r_est + row_idx * spacing_y
            
            # Determine x positions
            # If shift > 0, start at x = spacing_x/2 + r_est?
            # Let's just place centers at valid spots.
            
            # Simplified:
            # Row 0: 5 circles.
            # Row 1: 4 circles.
            # ...
            
            # Pattern for 26: 5, 4, 5, 4, 5, 3 (Total 26)
            pattern = [5, 4, 5, 4, 5, 3, 4, 5, ...] # Infinite
            # Actually 5+4+5+4+5+3 = 26.
            # 6 rows.
            
            row_counts = [5, 4, 5, 4, 5, 3]
            
            # Recalculate r_est based on fitting 6 rows
            # Height ~ 2*r + 5*spacing_y = 2r + 5*sqrt(3)r = r(2 + 8.66) = 10.66 r
            # r <= 1/10.66 ~ 0.094
            # Width for 5 circles: 2r + 4*2r = 10r <= 1 -> r <= 0.1
            # So r ~ 0.09 is safe.
            
            r_init = 0.09
            
            # Spacing
            dx = 2 * r_init
            dy = r_init * np.sqrt(3)
            
            # Positions
            # Row 0 (5 circles): x centered?
            # Width of 5 circles is 8*dx? No, 4 gaps of 2r. Width 8r.
            # Center of row at 0.5.
            # x positions: 0.5 - 4r, 0.5 - 2r, 0.5, 0.5 + 2r, 0.5 + 4r
            # Wait, 4r is 2*spacing? spacing is 2r.
            # So -4r, -2r, 0, 2r, 4r relative to center?
            # Total width 8r. Fits in 1.
            
            # Row 1 (4 circles): shifted by r (dx/2).
            # Width of 4 circles: 3 gaps of 2r -> 6r.
            # x positions relative to center: -3r, -r, r, 3r.
            
            # Let's implement this generically
            
            current_row_count = row_counts[row_idx] if row_idx < len(row_counts) else 0
            # If pattern exhausted, continue with last count?
            # For 26, 6 rows is enough.
            
            if row_idx < len(row_counts):
                 num_circles = row_counts[row_idx]
            else:
                 num_circles = row_counts[-1] # Fallback
            
            # But we need exactly n.
            # Let's just construct the list of centers based on a pattern.
            pass
            
    # Let's stick to a simple grid perturbed by random noise for robustness
    # The optimizer is strong.
    
    # Grid init
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Create a dense grid
    # Spacing 1/(cols+1) etc
    # But we want them closer to touch.
    # Let's estimate r.
    r_guess = 0.1
    # Place centers such that distance ~ 2*r_guess
    # Grid spacing
    gap = 1.0 / (cols + 1)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                x = gap * (c + 1)
                y = gap * (r + 1)
                # Perturb slightly to break symmetry
                x += (np.random.rand() - 0.5) * 0.01
                y += (np.random.rand() - 0.5) * 0.01
                centers[idx] = [x, y]
                idx += 1
                
    # Flatten
    params = np.zeros(3 * n)
    params[:2*n] = centers.flatten()
    params[2*n:] = radii
    
    return params

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    best_sum_radii = -1.0
    best_params = None
    
    # Try multiple restarts to find global optimum
    # We can use a few fixed seeds or random seeds
    seeds = [0, 1, 2, 42, 123, 100]
    
    # We can also try different initial geometries
    # 1. Random perturbation of grid
    # 2. Hexagonal pattern
    
    def try_optimization(seed, init_type='random'):
        np.random.seed(seed)
        
        if init_type == 'grid':
            params = create_initial_config(n, method='grid')
            # Perturb radii to be slightly larger to encourage expansion?
            # No, start small to be feasible.
            # But optimizer might start with invalid constraints if we start large.
            # 0.01 is safe.
        elif init_type == 'hex':
            # Hexagonal pattern
            params = np.zeros(3 * n)
            centers = np.zeros((n, 2))
            radii = np.ones(n) * 0.01
            
            # Pattern: 5, 4, 5, 4, 5, 3
            row_counts = [5, 4, 5, 4, 5, 3]
            r_init = 0.09 # Fits in square
            dx = 2 * r_init
            dy = r_init * np.sqrt(3)
            
            idx = 0
            for r_idx, count in enumerate(row_counts):
                if idx + count > n:
                    count = n - idx
                
                y = r_init + r_idx * dy
                
                # Shift
                shift = (dx / 2) if r_idx % 2 == 1 else 0
                
                # Center the row horizontally
                # Width of row = (count - 1) * dx
                # Start x = (1 - width) / 2 + r_init ?
                # Wait, r_init is radius.
                # If we place centers, we need to ensure they are within [0,1] roughly.
                # But optimizer will fix it.
                # Let's just place them centered.
                
                row_width = (count - 1) * dx
                x_start = 0.5 - row_width / 2 + shift # Center row, apply shift?
                # Actually, if we shift the row, the center shifts.
                # Let's just place them with some spacing.
                
                # Let's place centers at:
                # x = x_start + k * dx
                # But we want to center the whole cluster in [0,1].
                
                # Better: Calculate all x coords, then scale/translate to fit [0.1, 0.9]
                xs = []
                for k in range(count):
                    xs.append(shift + k * dx)
                
                # Center xs
                xs = np.array(xs)
                xs = xs - np.mean(xs)
                xs = xs * 0.8 # Shrink to fit
                xs = xs + 0.5
                
                ys = np.full(count, y)
                # Center ys
                # We have 6 rows. Total height ~ 5*dy + 2*r_init ~ 0.9.
                # Scale to fit.
                
                # Let's just set ys directly based on index
                # y = r_init + r_idx * dy
                # Scale y to be in middle.
                y_val = r_init + r_idx * dy
                
                for k in range(count):
                    if idx < n:
                        centers[idx] = [xs[k], y_val]
                        idx += 1
            
            # Scale centers to [0.1, 0.9] range to be safe
            centers = centers * 0.8 + 0.1
            
            params[:2*n] = centers.flatten()
            params[2*n:] = radii
            return params
        else:
            return None

        # Define bounds
        # x, y in [0, 1]
        # r in [0, 0.5] (max radius in unit square is 0.5)
        bounds = []
        for _ in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
        # Constraints
        # Inequality constraints g(x) >= 0
        cons = []
        
        # Overlap constraints
        # scipy.optimize expects dict or list of dicts
        # Nonlinear constraint
        cons.append({
            'type': 'ineq',
            'fun': constraints_overlap,
            'args': (n,)
        })
        
        # Boundary constraints
        cons.append({
            'type': 'ineq',
            'fun': constraints_boundary,
            'args': (n,)
        })
        
        # Optimization
        try:
            res = scipy.optimize.minimize(
                objective,
                params,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < 0: # fun is negative sum
                sum_r = -res.fun
                if sum_r > best_sum_radii:
                    best_sum_radii = sum_r
                    best_params = res.x
        except Exception as e:
            print(f"Optimization failed: {e}")
            pass

    # Run multiple trials
    # Try grid initialization
    for seed in seeds:
        try_optimization(seed, init_type='grid')
        
    # Try hex initialization
    for seed in seeds:
        try_optimization(seed, init_type='hex')
        
    # Extract result
    if best_params is None:
        # Fallback to simple grid
        params = create_initial_config(n, method='grid')
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        sum_radii = np.sum(radii)
        return centers, radii, sum_radii

    centers = best_params[:2*n].reshape(n, 2)
    radii = best_params[2*n:]
    
    # Clip radii just in case
    radii = np.clip(radii, 0, 0.5)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

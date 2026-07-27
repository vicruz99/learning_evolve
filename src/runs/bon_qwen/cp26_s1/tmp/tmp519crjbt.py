import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # Helper function to create initial configuration
    def get_initial_config(seed=0):
        rng = np.random.RandomState(seed)
        # Start with a grid-like distribution but slightly perturbed
        # 5x5 grid has 25 spots, we need 26.
        # Let's create a 6x5 grid pattern compressed or just random valid start
        
        # Strategy: Place centers in a grid that fits radius 0.05 safely
        # Grid spacing 0.2 fits radius 0.1? No, 0.2 spacing means dist=0.2, 2r=0.2 -> r=0.1.
        # If we have 26 circles, 5x5 is tight.
        # Let's try a 6x5 grid layout compressed into the square?
        # Or just random initialization with small radius and let optimizer expand.
        
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.05) # Start small
        
        # Generate a valid starting position to help optimizer
        # Simple grid 5x5 plus one extra
        idx = 0
        # 5 rows, 5 cols = 25
        # We can place the 26th in a gap or just perturb
        
        # Let's place them in a 5x6 grid pattern but scale to fit?
        # 6 cols width: (6-1)*spacing + 2*r <= 1 ?
        # Let's just use random positions in [0.1, 0.9]
        
        centers = rng.uniform(0.15, 0.85, size=(n, 2))
        # Ensure no initial overlap by simple projection or just rely on optimizer
        # With r=0.05, diameter 0.1. Random in 0.7x0.7 area.
        # Likely overlaps. 
        # Better: Grid
        rows = 6
        cols = 5
        # 6*5 = 30 slots, we pick 26
        # Spacing
        dx = 1.0 / (cols + 1)
        dy = 1.0 / (rows + 1)
        
        pos = 0
        for r in range(rows):
            for c in range(cols):
                if pos < n:
                    centers[pos] = [ (c+1)*dx, (r+1)*dy ]
                    pos += 1
                else:
                    break
            if pos >= n: break
        
        # Add some noise
        centers += rng.normal(0, 0.01, size=centers.shape)
        # Clamp
        centers = np.clip(centers, 0.1, 0.9)
        radii = np.full(n, 0.08)
        
        return centers, radii

    def objective(x_flat):
        # x_flat contains [x1, y1, r1, x2, y2, r2, ...]
        # We want to maximize sum of radii -> minimize negative sum
        r_vec = x_flat[2::3]
        return -np.sum(r_vec)

    # Constraints
    # 1. Boundary constraints for each circle
    # x >= r, x <= 1-r, y >= r, y <= 1-r
    # => x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    
    def get_constraints(x_flat):
        constraints = []
        
        # We can define constraints as functions that return value >= 0
        
        # Boundary constraints
        # For each circle i
        for i in range(n):
            idx_x = i * 3
            idx_y = i * 3 + 1
            idx_r = i * 3 + 2
            
            # x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': (lambda x, i=i: x[i*3] - x[i*3+2])
            })
            # 1 - x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': (lambda x, i=i: 1 - x[i*3] - x[i*3+2])
            })
            # y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': (lambda x, i=i: x[i*3+1] - x[i*3+2])
            })
            # 1 - y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': (lambda x, i=i: 1 - x[i*3+1] - x[i*3+2])
            })
            # r >= 0 (non-negativity) - usually handled by bounds, but let's be safe
            # Actually SLSQP supports bounds.
            
        # Pairwise non-overlap constraints
        # dist(i, j) >= r_i + r_j
        # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
        # Squared form might be easier? 
        # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        # But sqrt is smooth. Let's use sqrt.
        
        for i in range(n):
            for j in range(i + 1, n):
                idx_xi, idx_yi, idx_ri = i*3, i*3+1, i*3+2
                idx_xj, idx_yj, idx_rj = j*3, j*3+1, j*3+2
                
                def overlap_constraint(x, i=i, j=j):
                    xi, yi, ri = x[idx_xi], x[idx_yi], x[idx_ri]
                    xj, yj, rj = x[idx_xj], x[idx_yj], x[idx_rj]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return dist - (ri + rj)
                
                constraints.append({
                    'type': 'ineq',
                    'fun': overlap_constraint
                })
        
        return constraints

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 1] (loose bound)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 1)]) # x, y, r

    best_sum = 0
    best_centers = None
    best_radii = None

    # Try multiple restarts
    for seed in range(10):
        centers, radii = get_initial_config(seed)
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[i*3] = centers[i, 0]
            x0[i*3+1] = centers[i, 1]
            x0[i*3+2] = radii[i]

        try:
            # SLSQP can handle many constraints, but it might be slow.
            # We can reduce constraints by assuming equal radii initially?
            # But we want variable radii.
            # To speed up, maybe fix radii to be equal in first pass?
            # Let's try full optimization but maybe limit iterations or use simpler method if too slow.
            # However, 26 circles is small enough.
            
            cons = get_constraints(x0)
            
            # Use SLSQP
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                               options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[0::3].reshape(n, 2)
                    best_radii = res.x[2::3]
            else:
                # Even if not successful, check if valid and better
                # But usually we want valid solution.
                pass
        except Exception as e:
            print(f"Optimization failed for seed {seed}: {e}")
            continue

    # Post-processing: Ensure strict validity and maybe slight shrinking if on edge
    if best_centers is not None:
        # Check overlaps and fix if necessary (very tight margins)
        # The validator allows 1e-12 tolerance, so we should be fine.
        # But let's ensure radii are not negative and centers valid.
        
        # A small scaling down factor to be safe against numerical noise?
        # No, validator has tolerance.
        
        # Just return the best found
        # Ensure shapes
        return best_centers, best_radii, float(best_sum)

    # Fallback: simple grid packing if optimizer fails completely
    # 5x5 grid r=0.1, 26th circle small?
    # This shouldn't happen.
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    idx = 0
    for r in range(5):
        for c in range(5):
            if idx < n:
                centers[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                radii[idx] = 0.1
                idx += 1
    # Place 26th circle
    if idx < n:
        centers[idx] = [0.5, 0.5] # Overlaps, but better to have something
        radii[idx] = 0.01 # Small to avoid massive overlap? 
    return centers, radii, 0.0
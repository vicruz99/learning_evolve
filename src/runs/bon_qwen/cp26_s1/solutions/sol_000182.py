# sol_000182 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9afef83a) state=3eca54b1 sum of radii=2.592680 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, differential_evolution
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        if np.isnan(radii[i]):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Helper to define constraints for the optimizer
    def constraints_builder(centers_flat, radii_flat):
        centers = centers_flat.reshape(-1, 2)
        radii = radii_flat
        
        cons = []
        
        # 1. Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # x - r >= 0  => x - r >= 0
        # 1 - x - r >= 0 => 1 - x - r >= 0
        # Same for y
        
        for i in range(n_circles):
            x = centers[i, 0]
            y = centers[i, 1]
            r = radii[i]
            
            # x >= r
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[2*idx] - v[2*n_circles + idx]}) # x_i - r_i
            # 1 - x - r >= 0 => x + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1 - v[2*idx] - v[2*n_circles + idx]})
            # y >= r
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[2*idx + 1] - v[2*n_circles + idx]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1 - v[2*idx + 1] - v[2*n_circles + idx]})
            
            # r >= 0 (handled by bounds mostly, but good to have)
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[2*n_circles + idx]})

        # 2. Non-overlap constraints: dist_ij >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                def make_constraint(idx_i, idx_j):
                    def constraint(v):
                        xi = v[2*idx_i]
                        yi = v[2*idx_i + 1]
                        ri = v[2*n_circles + idx_i]
                        
                        xj = v[2*idx_j]
                        yj = v[2*idx_j + 1]
                        rj = v[2*n_circles + idx_j]
                        
                        dist_sq = (xi - xj)**2 + (yi - yj)**2
                        sum_r_sq = (ri + rj)**2
                        return dist_sq - sum_r_sq
                    return constraint
                
                cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})
                
        return cons

    # Objective: Maximize sum(radii) => Minimize -sum(radii)
    def objective(v):
        radii = v[2*n_circles:]
        return -np.sum(radii)

    bounds = []
    for i in range(2 * n_circles):
        bounds.append((0.0, 1.0))
    for i in range(n_circles):
        bounds.append((0.0, 0.5)) # Upper bound for radius, 0.5 is safe max

    # We will run a few trials with different initializations
    # Trial 1: Hexagonal packing approximation
    # Trial 2: Random initialization
    # Trial 3: Perturbed grid
    
    trials = []

    # 1. Hexagonal Packing
    # Rows of 5, 5, 5, 5, 6? 
    # To fit 26 circles, maybe 5 rows with varying counts.
    # Let's try to pack them as densely as possible.
    # A hexagonal lattice has spacing 2r. Vertical spacing sqrt(3)r.
    # Let's assume r ~ 0.1.
    
    def generate_hex_init():
        # Try to fit 26 circles in a hexagonal pattern
        # Let's aim for 5 rows. 5+5+5+5+6 = 26.
        # Or 6 rows? 4+5+4+5+4+4 = 26?
        # Let's try a dense cluster.
        
        # Heuristic: Place centers in a triangular grid
        # Grid points (i, j) -> x = i*dx + (j%2)*dx/2, y = j*dy
        # dx = 2r, dy = sqrt(3)r
        # We don't know r yet, let's assume r=0.1 for initialization positions
        r_init = 0.1
        dx = 2 * r_init
        dy = math.sqrt(3) * r_init
        
        centers = []
        # Try to fill the square
        # We can just iterate grid points and pick 26 closest to center or just first 26
        # But we need them inside [r, 1-r] approx.
        # Let's generate a large grid and select valid ones
        
        candidates = []
        for j in range(10):
            for i in range(10):
                x = i * dx + (j % 2) * (dx / 2)
                y = j * dy
                # Shift to center
                x += 0.05 
                y += 0.05
                if 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9:
                    candidates.append([x, y])
        
        # If we don't have enough, add more by relaxing bounds slightly or adding random
        while len(candidates) < 26:
            candidates.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
            
        selected_centers = candidates[:26]
        radii = np.full(26, 0.08) # Start small to be valid
        return selected_centers, radii

    # 2. Random Initialization
    def generate_random_init():
        centers = np.random.uniform(0.1, 0.9, size=(26, 2))
        radii = np.full(26, 0.02)
        return centers, radii

    # 3. Grid Initialization (5x5 + 1)
    def generate_grid_init():
        centers = []
        # 5x5 grid points
        pts = [0.2, 0.4, 0.6, 0.8, 1.0] # Wait, 1.0 is edge. 
        # Let's use 0.1, 0.3, 0.5, 0.7, 0.9
        pts = [0.1, 0.3, 0.5, 0.7, 0.9]
        for y in pts:
            for x in pts:
                centers.append([x, y])
                if len(centers) == 25:
                    break
            if len(centers) == 25:
                break
        # Add 26th at center or random gap
        centers.append([0.5, 0.5]) # Might overlap heavily
        # Better: push out or just random
        centers[-1] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
        
        radii = np.full(26, 0.08)
        return centers, radii

    init_funcs = [generate_hex_init, generate_random_init, generate_grid_init]
    
    for func in init_funcs:
        try:
            centers, radii = func()
            v0 = np.concatenate([centers.flatten(), radii])
            
            # Run optimization
            # SLSQP can be sensitive to initial constraints. 
            # If v0 violates constraints, it might fail.
            # Our r=0.08 or 0.02 should be safe if centers are spread.
            
            # To be safe, let's ensure v0 is valid or at least not terrible.
            # With r=0.02, almost any placement is valid.
            
            cons = constraints_builder(v0, radii) # This builder creates closures, need to pass v0? 
            # Actually the builder uses n_circles from outer scope, but the lambdas capture i, j.
            # The 'fun' in constraints needs to accept 'v'.
            
            # Re-define constraints properly for the solver call
            # The builder above creates a list of dicts. The funs are already bound.
            # But I need to pass the list to minimize.
            
            # Wait, the builder in the code block above creates constraints that depend on v.
            # But I called it with v0? No, I just defined the builder logic.
            # Let's rewrite constraint generation to be clean.
            
            constraints = []
            # Boundary
            for i in range(n_circles):
                # x >= r => x - r >= 0
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n_circles + i]})
                # 1 - x - r >= 0
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i] - v[2*n_circles + i]})
                # y >= r
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i + 1] - v[2*n_circles + i]})
                # 1 - y - r >= 0
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i + 1] - v[2*n_circles + i]})
                # r >= 0
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*n_circles + i]})
            
            # Non-overlap
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    constraints.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: 
                             (v[2*i] - v[2*j])**2 + (v[2*i + 1] - v[2*j + 1])**2 - (v[2*n_circles + i] + v[2*n_circles + j])**2
                    })
            
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or (res.fun < 0 and validate_packing(res.x[:2*n_circles].reshape(-1, 2), res.x[2*n_circles:])):
                curr_sum = -res.fun
                if curr_sum > best_sum_radii:
                    best_sum_radii = curr_sum
                    best_centers = res.x[:2*n_circles].reshape(-1, 2)
                    best_radii = res.x[2*n_circles:]
                    
        except Exception as e:
            pass

    # If best is still low, try a specific heuristic for 26 circles
    # Maybe the optimizer got stuck.
    # Let's try to refine the best solution found by a second pass with tighter bounds or different method?
    # Or just return the best.
    
    # Ensure output is valid
    if best_centers is not None:
        # Double check validity
        if not validate_packing(best_centers, best_radii):
            # If invalid, fallback to a known valid grid
            # 5x5 grid r=0.1, sum=2.5. Plus one tiny circle?
            # Let's construct a valid 26 circle packing manually as fallback
            pts = [0.1, 0.3, 0.5, 0.7, 0.9]
            centers_fallback = []
            radii_fallback = []
            for y in pts:
                for x in pts:
                    centers_fallback.append([x, y])
                    radii_fallback.append(0.1)
                    if len(centers_fallback) == 25: break
                if len(centers_fallback) == 25: break
            # Add 26th circle in a gap?
            # Gaps are at (0.2, 0.2) etc. Distance to (0.1,0.1) is 0.1414.
            # r_new + 0.1 <= 0.1414 => r_new <= 0.0414
            centers_fallback.append([0.2, 0.2])
            radii_fallback.append(0.041)
            
            best_centers = np.array(centers_fallback)
            best_radii = np.array(radii_fallback)
            best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum_radii)

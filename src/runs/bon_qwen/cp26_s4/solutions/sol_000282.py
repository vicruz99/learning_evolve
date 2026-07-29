# sol_000282 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 20c39dac) state=886955a5 sum of radii=2.541400 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
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
        elif np.isnan(radii[i]):
            return False
    for i in range(n):
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

def objective(vars, n):
    # vars contains x, y, r for each circle interleaved or structured?
    # Let's use structure: [x1, y1, r1, x2, y2, r2, ...]
    # Actually easier to separate in constraint functions, but minimize needs 1D array.
    # We will access r_i directly.
    # vars layout: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    r_sum = 0
    for i in range(n):
        r_sum += vars[3*i + 2]
    return -r_sum  # Minimize negative sum

def get_constraints(n):
    constraints = []
    
    # Boundary constraints
    # x_i - r_i >= 0  => r_i - x_i <= 0
    # x_i + r_i <= 1  => x_i + r_i - 1 <= 0
    # y_i - r_i >= 0  => r_i - y_i <= 0
    # y_i + r_i <= 1  => y_i + r_i - 1 <= 0
    
    for i in range(n):
        idx_x = 3*i
        idx_y = 3*i + 1
        idx_r = 3*i + 2
        
        # x_i >= r_i
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        })
        # x_i <= 1 - r_i  => x_i + r_i <= 1 => 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[idx_x] - v[idx_r]
        })
        # y_i >= r_i
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        })
        # y_i <= 1 - r_i
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[idx_y] - v[idx_r]
        })

    # Overlap constraints
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3*i, 3*i+1, 3*i+2
            idx_xj, idx_yj, idx_rj = 3*j, 3*j+1, 3*j+2
            
            def fun(v, i=i, j=j):
                dx = v[idx_xi] - v[idx_xj]
                dy = v[idx_yi] - v[idx_yj]
                dr = v[idx_ri] + v[idx_rj]
                return (dx*dx + dy*dy) - (dr*dr)
            
            constraints.append({'type': 'ineq', 'fun': fun})
            
    return constraints

def generate_initial_config(n, method='grid'):
    # Returns array of shape (n, 3) [x, y, r]
    config = np.zeros((n, 3))
    
    if method == 'grid':
        # 5x5 grid is 25 circles. Add 1 in center gap.
        # Grid spacing 0.2. Centers at 0.1, 0.3, 0.5, 0.7, 0.9
        # Radius 0.09 initially to allow movement.
        r_init = 0.09
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < n:
                    x = 0.1 + c * 0.2
                    y = 0.1 + r * 0.2
                    config[idx] = [x, y, r_init]
                    idx += 1
        
        # Add 26th circle
        if idx < n:
            # Place in a gap, e.g., center of a cell (0.2, 0.2)
            # Or random small perturbation
            config[idx] = [0.2, 0.2, 0.04]
            idx += 1
            
    elif method == 'random':
        for i in range(n):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.05, 0.12)
            config[i] = [x, y, r]
            
    elif method == 'hexagonal':
        # Hexagonal packing attempt
        # 5 rows
        # Row 0: 6 circles? Width constraint might be tight.
        # Let's try 6, 5, 6, 5, 4 pattern or similar.
        # Actually, let's just generate points in a hex lattice and pick 26.
        pts = []
        s = 0.2 # approximate spacing
        for row in range(6):
            y = 0.1 + row * (s * math.sqrt(3)/2)
            offset = (row % 2) * (s/2)
            x = 0.1 + offset
            while x <= 0.9:
                pts.append([x, y])
                x += s
            if len(pts) >= n:
                break
        
        # Fill config
        for i in range(n):
            if i < len(pts):
                config[i] = [pts[i][0], pts[i][1], 0.08]
            else:
                config[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05]

    return config.flatten()

def run_packing():
    n = 26
    constraints = get_constraints(n)
    
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Try multiple initializations
    methods = ['grid', 'hexagonal', 'random', 'random']
    
    for method in methods:
        # Perturb grid slightly
        if method == 'grid':
            init_vars = generate_initial_config(n, 'grid')
            # Add small noise
            noise = np.random.normal(0, 0.02, size=init_vars.shape)
            init_vars = np.clip(init_vars + noise, 0.01, 0.99) # Keep r > 0
        else:
            init_vars = generate_initial_config(n, method)
            
        # Bounds for variables
        # x, y in [0, 1]
        # r in [0, 0.5] (max possible radius in unit square is 0.5)
        bounds = [(0, 1)] * (3 * n)
        for i in range(n):
            bounds[3*i + 2] = (0, 0.5) # radius bound

        try:
            res = opt.minimize(
                objective,
                init_vars,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    # Validate before accepting
                    c_opt = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
                    r_opt = np.array([res.x[3*i+2] for i in range(n)])
                    
                    # Numerical validation with tolerance
                    valid = True
                    # Check NaN
                    if np.isnan(c_opt).any() or np.isnan(r_opt).any():
                        valid = False
                    # Check bounds
                    for i in range(n):
                        if r_opt[i] < 0: valid = False
                        x, y = c_opt[i]
                        r = r_opt[i]
                        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                            valid = False
                    # Check overlap
                    for i in range(n):
                        for j in range(i + 1, n):
                            dist = np.sqrt(np.sum((c_opt[i] - c_opt[j])**2))
                            if dist < r_opt[i] + r_opt[j] - 1e-9:
                                valid = False
                    
                    if valid:
                        best_sum = current_sum
                        best_centers = c_opt
                        best_radii = r_opt
        except Exception:
            continue

    # If no valid solution found (unlikely with grid init), return a valid baseline
    if best_centers is None:
        # Fallback to 5x5 grid + 1 small
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        idx = 0
        for r in range(5):
            for c in range(5):
                centers[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                radii[idx] = 0.1
                idx += 1
        centers[25] = [0.2, 0.2]
        radii[25] = 0.0414
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)

    # Final check and cleanup (clip small negative radii if any due to float error)
    best_radii = np.maximum(best_radii, 0)
    
    return best_centers, best_radii, float(best_sum)

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")
    # Check if close to target
    if s > 2.635:
        print("Target reached!")
    else:
        print(f"Gap to target: {2.636 - s}")

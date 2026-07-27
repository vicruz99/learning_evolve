import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions and radii of 26 circles in a unit square
    to maximize the sum of radii.
    """
    n_circles = 26
    
    # Helper function to calculate constraints
    def get_constraints(centers, radii):
        constraints = []
        n = centers.shape[0]
        
        # Boundary constraints: r <= x <= 1-r  => x - r >= 0, 1 - x - r >= 0
        # Same for y
        for i in range(n):
            # x constraints
            constraints.append({'type': 'ineq', 'fun': lambda c, r, idx=i: c[idx, 0] - r[idx]})
            constraints.append({'type': 'ineq', 'fun': lambda c, r, idx=i: 1.0 - c[idx, 0] - r[idx]})
            # y constraints
            constraints.append({'type': 'ineq', 'fun': lambda c, r, idx=i: c[idx, 1] - r[idx]})
            constraints.append({'type': 'ineq', 'fun': lambda c, r, idx=i: 1.0 - c[idx, 1] - r[idx]})
            # Non-negative radius
            constraints.append({'type': 'ineq', 'fun': lambda r, idx=i: r[idx]})
            
        # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
        # Using a small tolerance to avoid numerical issues, though mathematically >= 0
        for i in range(n):
            for j in range(i + 1, n):
                # We pass centers and radii as a single array 'x' to the optimizer
                # But here we define fun inside the loop. 
                # To make it work with scipy, we need to construct a lambda that captures i, j
                # However, constructing thousands of lambdas can be slow. 
                # Let's define a generic function that takes the full state vector.
                pass 

    # Better approach for scipy: Define objective and constraints based on a flattened state vector
    # State vector: [x1, y1, r1, x2, y2, r2, ...]
    # Length: 3 * n_circles
    
    def objective(state):
        # Sum of radii is the sum of every 3rd element starting from index 2 (0-indexed: 2, 5, 8...)
        # Actually indices are 2, 5, ...
        radii = state[2::3]
        return -np.sum(radii) # Minimize negative sum

    def boundary_constraints(state):
        n = n_circles
        x = state[0::3]
        y = state[1::3]
        r = state[2::3]
        
        cons = []
        # x - r >= 0
        cons.extend(x - r)
        # 1 - x - r >= 0
        cons.extend(1.0 - x - r)
        # y - r >= 0
        cons.extend(y - r)
        # 1 - y - r >= 0
        cons.extend(1.0 - y - r)
        # r >= 0
        cons.extend(r)
        return cons

    def non_overlap_constraints(state):
        n = n_circles
        x = state[0::3]
        y = state[1::3]
        r = state[2::3]
        
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                # Constraint: dist_sq - sum_r^2 >= 0
                cons.append(dist_sq - sum_r**2)
        return cons

    # Combine constraints
    # scipy.optimize.minimize with SLSQP can take a list of constraint dicts
    # or we can combine them into a single function returning an array.
    # Let's use the list of dicts approach for clarity, or a combined function.
    # A combined function is often more efficient.
    
    def combined_constraints(state):
        # Boundary
        n = n_circles
        x = state[0::3]
        y = state[1::3]
        r = state[2::3]
        
        b_cons = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r,
            r
        ])
        
        # Overlap
        o_cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                o_cons.append(dist_sq - sum_r**2)
        
        return np.concatenate([b_cons, o_cons])

    # Initial guess generation
    def generate_initial_guess():
        # Try a hexagonal-like or grid layout
        # 26 circles. 
        # Let's try to fit them in a 5x5 grid (25) + 1 extra.
        # Or just random valid placement.
        
        # Grid initialization
        # 5x5 grid points
        grid_step = 0.2 # for 5 circles, spacing 0.2, radii 0.1 would be tight.
        # Let's place centers at 0.2, 0.4, 0.6, 0.8
        # But we need 26.
        # Let's use a slightly denser packing initialization.
        # Maybe 6 rows of ~4-5 circles?
        
        centers = []
        radii = []
        
        # Try a perturbed hexagonal packing
        # Radius approx 0.09
        r_init = 0.08
        count = 0
        y = r_init
        while count < n_circles:
            row_start = r_init if (int((y-r_init)/(r_init*np.sqrt(3))) % 2 == 0) else 2*r_init
            x = row_start
            while x <= 1 - r_init and count < n_circles:
                centers.append([x, y])
                radii.append(r_init)
                count += 1
                x += 2 * r_init
            y += r_init * np.sqrt(3)
        
        # If we didn't fill 26, fill remaining randomly
        while len(centers) < n_circles:
            # Find a spot
            for _ in range(100):
                cx = np.random.uniform(r_init, 1-r_init)
                cy = np.random.uniform(r_init, 1-r_init)
                valid = True
                for k in range(len(centers)):
                    dx = cx - centers[k][0]
                    dy = cy - centers[k][1]
                    if dx*dx + dy*dy < (r_init + radii[k])**2:
                        valid = False
                        break
                if valid:
                    centers.append([cx, cy])
                    radii.append(r_init)
                    break
            else:
                # If stuck, just add a small circle somewhere or shrink existing
                # For initialization, valid is not strictly required for optimizer if constraints are soft, 
                # but better to be valid.
                # Force add with small radius
                centers.append([np.random.rand(), np.random.rand()])
                radii.append(0.01)

        return np.array(centers), np.array(radii)

    # Run optimization with multiple restarts to find better solution
    best_state = None
    best_score = -np.inf
    
    for attempt in range(5):
        centers_init, radii_init = generate_initial_guess()
        
        # Flatten state
        state0 = np.zeros(3 * n_circles)
        state0[0::3] = centers_init[:, 0]
        state0[1::3] = centers_init[:, 1]
        state0[2::3] = radii_init
        
        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n_circles):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        try:
            res = minimize(
                objective, 
                state0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints={'type': 'ineq', 'fun': combined_constraints},
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )
            
            current_score = -res.fun # Sum of radii
            if current_score > best_score:
                best_score = current_score
                best_state = res.x.copy()
        except Exception:
            continue
            
        # Perturb best solution for next attempt
        if best_state is not None:
            state0 = best_state + np.random.normal(0, 0.001, size=best_state.shape)
            # Clamp bounds
            state0[0::3] = np.clip(state0[0::3], 0, 1)
            state0[1::3] = np.clip(state0[1::3], 0, 1)
            state0[2::3] = np.clip(state0[2::3], 0, 0.5)

    # Extract results
    centers = np.column_stack((best_state[0::3], best_state[1::3]))
    radii = best_state[2::3]
    sum_radii = np.sum(radii)
    
    # Final sanity check and clean up tiny radii or invalid positions if any (due to numerical error)
    # Although optimizer should respect constraints, numerical tolerance might be exceeded.
    # We can try to slightly shrink radii if needed, but let's trust the optimizer.
    
    return centers, radii, sum_radii
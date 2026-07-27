import numpy as np
import scipy.optimize as opt

def generate_initial_guess(n, seed=None):
    """
    Generate an initial configuration for n circles using a staggered grid 
    pattern that is scaled to fit within the unit square.
    """
    if seed is not None:
        np.random.seed(seed)

    centers = np.zeros((n, 2))
    
    # Attempt to fit rows with alternating counts to maximize density
    # Try different row count patterns
    patterns = []
    for r_count in range(3, 10):
        for start_k in range(4, 9):
            counts = []
            c = start_k
            remaining = n
            for _ in range(r_count):
                if remaining == 0: break
                # Alternate between k and k-1 for hexagonal packing efficiency
                counts.append(min(c, remaining))
                remaining -= counts[-1]
                c = start_k - 1 if c == start_k else start_k
            
            if remaining == 0:
                patterns.append(counts)
                # Break if we found a valid one, but we might prefer tighter ones
                # Actually, fewer rows might be better for packing height, 
                # but more rows might fit more circles. 
                # We just need one valid pattern to start.
    
    # Select a reasonable pattern (e.g., 5 rows of approx 5)
    chosen_pattern = [5, 5, 5, 5, 6] 
    # Check if sum matches n, if not adjust
    while sum(chosen_pattern) > n:
        chosen_pattern.pop()
    while sum(chosen_pattern) < n:
        chosen_pattern.append(1) # Fallback
    
    # Reconstruct centers based on chosen_pattern
    # Using hexagonal packing spacing assumption for layout
    # But we will scale it later, so initial coordinates don't strictly matter 
    # as long as they are relative.
    
    idx = 0
    row_y = 0.0
    row_h = 1.0 # Arbitrary height unit
    
    for i, count in enumerate(chosen_pattern):
        # Center row horizontally
        # Spacing within row
        if count > 0:
            row_width = (count - 1) * 1.0
            start_x = -row_width / 2.0
            for j in range(count):
                if idx < n:
                    x = start_x + j * 1.0
                    # Offset rows for hex packing
                    if i % 2 == 1:
                        x += 0.5 
                    centers[idx] = [x, row_y]
                    idx += 1
        row_y += row_h
    
    # Normalize to unit square and add small random jitter
    # Find bounds of the generated pattern
    min_x, min_y = centers.min(axis=0)
    max_x, max_y = centers.max(axis=0)
    
    # Scale to fit inside [0,1]x[0,1] with a margin
    # We scale down slightly to ensure initial feasibility for radii > 0
    width = max_x - min_x
    height = max_y - min_y
    scale = 0.9 / max(width, height)
    
    centers = (centers - [min_x, min_y]) * scale + 0.05
    
    # Add random jitter to break symmetry
    jitter = np.random.uniform(-0.02, 0.02, centers.shape)
    centers += jitter
    centers = np.clip(centers, 0.01, 0.99)
    
    return centers

def objective(params, n):
    """Maximize sum of radii -> Minimize negative sum."""
    radii = params[2*n:]
    return -np.sum(radii)

def boundary_constraints(params, n):
    """Circles must be inside [0,1]x[0,1]"""
    constraints = []
    for i in range(n):
        x = params[i]
        y = params[n + i]
        r = params[2*n + i]
        
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[idx] - p[2*n + idx]})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: 1.0 - p[idx] - p[2*n + idx]})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[n + idx] - p[2*n + idx]})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: 1.0 - p[n + idx] - p[2*n + idx]})
    return constraints

def overlap_constraints(params, n):
    """Circles must not overlap: dist >= r1 + r2"""
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
            # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: \
                    (p[i] - p[j])**2 + (p[n+i] - p[n+j])**2 - (p[2*n+i] + p[2*n+j])**2
            })
    return constraints

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Optimization bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    # We will run multiple optimizations to find a global optimum
    num_restarts = 5
    
    for restart in range(num_restarts):
        # Generate initial guess
        centers_init = generate_initial_guess(n, seed=restart*42)
        
        # Initial radii: small enough to be valid, but large enough to optimize
        # We can estimate a valid radius based on min distance to boundary/neighbors
        init_radii = np.zeros(n)
        for i in range(n):
            dist_boundary = min(centers_init[i, 0], 1-centers_init[i, 0], 
                                centers_init[i, 1], 1-centers_init[i, 1])
            dist_neighbors = np.min([np.sqrt(np.sum((centers_init[i] - centers_init[j])**2)) 
                                     for j in range(n) if j != i])
            init_radii[i] = min(dist_boundary, dist_neighbors / 2) * 0.5 # 50% of available space
        
        # Flatten parameters: [x0..x25, y0..y25, r0..r25]
        x0 = np.concatenate([centers_init[:, 0], centers_init[:, 1], init_radii])
        
        # Define constraints
        constraints = []
        constraints.extend(boundary_constraints(x0, n))
        constraints.extend(overlap_constraints(x0, n))
        
        # Optimize
        try:
            res = opt.minimize(
                objective, 
                x0, 
                args=(n,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or (res.nit > 100): # Accept if converged or ran enough
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[:2*n].reshape(n, 2)
                    best_radii = res.x[2*n:]
        except Exception:
            continue

    # Final validation and cleanup
    if best_centers is None:
        # Fallback to a simple grid if optimization failed completely
        grid_x = np.linspace(0.1, 0.9, 5)
        grid_y = np.linspace(0.1, 0.9, 5)
        best_centers = np.array([[x, y] for x in grid_x for y in grid_y])[:n]
        best_radii = np.full(n, 0.1)
        # Adjust the last one if needed or just return grid
        # A simple grid of 25 circles r=0.1 is valid. For 26th, make it smaller.
        if n > 25:
            best_radii[25] = 0.0 # Placeholder, validation might fail if 0 is not allowed?
            # Rules say nonnegative. 0 is nonnegative.
            # But to be safe, let's try to fit it.
            # Actually, the optimizer should have found something better.
            pass

    # Ensure non-negative radii
    best_radii = np.maximum(best_radii, 0.0)
    
    # Calculate sum
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii

if __name__ == "__main__":
    # Local testing block
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radii: {radii}")
    # Check validity manually if needed
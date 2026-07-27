import numpy as np
from scipy.optimize import minimize

def run_packing():
    N = 26
    
    # Smooth penalty function for better gradient-based optimization
    # Use log-sum-exp or just squared hinge loss which is piecewise smooth
    # Hinge loss squared is not smooth at 0, but usually optimizers handle it.
    # Let's try to use a smooth approximation: softplus squared?
    # softplus(x) = log(1 + exp(x))
    # penalty ~ (softplus(k * violation))^2
    # For large k, this approximates max(0, violation)^2.
    
    k_smooth = 10.0
    
    def smooth_max(val):
        # Approximation of max(0, val)
        # Using log(1 + exp(val)) / k ? No, log(1+exp(val)) approx max(0, val) + ln(2)?
        # Actually log(1+exp(x)) is smooth.
        # To approximate max(0,x), we can use (x + sqrt(x^2 + eps))/2 ?
        # Or just rely on standard max(0, val)**2 which is convex and L-smooth.
        # scipy optimizers often handle it.
        if val > 0:
            return val
        return 0.0
    
    def compute_cost(params_vec):
        centers_radii = params_vec.reshape((N, 3))
        centers = centers_radii[:, :2]
        radii = centers_radii[:, 2]
        
        cost = -np.sum(radii)
        violation_sum = 0.0
        
        penalty_weight = 5000.0
        
        # Boundary violations
        for i in range(N):
            x, y, r = centers_radii[i]
            # x >= r => r - x <= 0. Violation r - x.
            # x <= 1-r => x + r - 1 <= 0. Violation x + r - 1.
            v1 = smooth_max(r - x)
            v2 = smooth_max(x + r - 1)
            v3 = smooth_max(r - y)
            v4 = smooth_max(y + r - 1)
            violation_sum += (v1**2 + v2**2 + v3**2 + v4**2)
            
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers_radii[i, 0] - centers_radii[j, 0]
                dy = centers_radii[i, 1] - centers_radii[j, 1]
                # dist = sqrt(dx^2 + dy^2)
                # To avoid sqrt and issues with 0, we can compare squared distances?
                # Constraint: dist >= r_i + r_j
                # <=> dist^2 >= (r_i + r_j)^2
                # But smooth max works better on linear-ish constraints.
                # Let's stick to dist.
                dist = np.hypot(dx, dy)
                min_dist = radii[i] + radii[j]
                v = smooth_max(min_dist - dist)
                violation_sum += v**2
                
        cost += penalty_weight * violation_sum
        return cost

    def compute_cost_smooth(params_vec):
        # Using a smooth approximation for max to help gradient methods
        centers_radii = params_vec.reshape((N, 3))
        centers = centers_radii[:, :2]
        radii = centers_radii[:, 2]
        
        cost = -np.sum(radii)
        violation_sum = 0.0
        
        penalty_weight = 2000.0
        eps = 1e-4
        
        # Boundary
        for i in range(N):
            x, y, r = centers_radii[i]
            # Violations: r - x, x + r - 1, r - y, y + r - 1
            vals = [r - x, x + r - 1, r - y, y + r - 1]
            for v in vals:
                # Smooth max: (v + sqrt(v^2 + eps))/2 approx max(0, v) ?
                # Actually log(1 + exp(v/k)) * k is a smooth max.
                # Let's use simple squared max for now, but check gradient.
                if v > 0:
                    violation_sum += v**2
        
        # Overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers_radii[i, 0] - centers_radii[j, 0]
                dy = centers_radii[i, 1] - centers_radii[j, 1]
                dist = np.hypot(dx, dy)
                min_dist = radii[i] + radii[j]
                v = min_dist - dist
                if v > 0:
                    violation_sum += v**2
                    
        cost += penalty_weight * violation_sum
        return cost

    # Initialization: Hexagonal grid
    r_init = 0.105
    init_params = []
    y = r_init
    row_idx = 0
    circles_placed = 0
    
    # Try to pack tightly
    row_height = r_init * np.sqrt(3)
    
    while circles_placed < N:
        if row_idx % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init
            
        x = x_start
        while x <= 1 - r_init and circles_placed < N:
            init_params.extend([x, y, r_init])
            circles_placed += 1
            x += 2 * r_init
        
        y += row_height
        row_idx += 1
        
    # Fill remaining if any
    while len(init_params) < 3 * N:
        init_params.extend([0.5, 0.5, 0.01])
    
    init_params = np.array(init_params[:3*N])
    
    best_cost = float('inf')
    best_params = init_params
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    # Run optimization
    # Use L-BFGS-B as it respects bounds
    for _ in range(3): # Few restarts
        # Add noise
        x0 = init_params + np.random.normal(0, 0.01, size=init_params.shape)
        # Clip bounds for x0
        x0 = np.clip(x0, [0, 0, 0], [1, 1, 0.5])
        
        res = minimize(compute_cost_smooth, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
            
    # Extract
    params = best_params.reshape((N, 3))
    centers = params[:, :2].copy()
    radii = params[:, 2].copy()
    
    # Ensure non-negative radii
    radii = np.maximum(radii, 1e-7)
    
    # Clip centers to valid region based on radii
    for i in range(N):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
        
    # Final check and slight shrinkage if overlaps persist due to optimization tolerance
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            req = radii[i] + radii[j]
            if dist < req - 1e-9:
                # Overlap
                overlap = req - dist
                # Reduce radii to fix overlap?
                # Simple strategy: scale down both radii by overlap/2?
                # Or just scale down radii[i] and radii[j]
                reduction = overlap / 2
                radii[i] = max(radii[i] - reduction, 0)
                radii[j] = max(radii[j] - reduction, 0)
                # Recalculate bounds
                centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
                centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])
                centers[j, 0] = np.clip(centers[j, 0], radii[j], 1 - radii[j])
                centers[j, 1] = np.clip(centers[j, 1], radii[j], 1 - radii[j])

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
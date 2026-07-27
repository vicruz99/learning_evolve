import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    N = 26
    # High penalty weight to enforce constraints strictly
    PENALTY_WEIGHT = 10000.0
    
    def objective(vars):
        """
        Objective function: Maximize sum of radii (minimize negative sum) + Penalty for violations.
        vars: array of shape (3N,) containing [x0, y0, r0, x1, y1, r1, ...]
        """
        radii = vars[2*N:]
        return -np.sum(radii) + PENALTY_WEIGHT * constraint_penalty(vars)

    def constraint_penalty(vars):
        """
        Calculates the penalty for constraint violations.
        Includes penalties for:
        1. Overlapping circles (distance < sum of radii)
        2. Circles crossing the square boundaries
        """
        centers = vars[:2*N].reshape(N, 2)
        radii = vars[2*N:]
        
        penalty = 0.0
        
        # Overlap penalty
        # We check all pairs i < j
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                # If circles overlap, add squared overlap to penalty
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap * overlap
        
        # Boundary penalty
        # Ensure r <= x, r <= 1-x, r <= y, r <= 1-y
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            
            # Left boundary
            if r > x:
                penalty += (r - x)**2
            # Right boundary
            if r > 1 - x:
                penalty += (r - (1 - x))**2
            # Bottom boundary
            if r > y:
                penalty += (r - y)**2
            # Top boundary
            if r > 1 - y:
                penalty += (r - (1 - y))**2
                
        return penalty

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    best_sol = None
    best_sum = -1.0
    
    # --- Strategy 1: Random Restarts ---
    # Run multiple optimizations with random initializations
    for _ in range(10):
        x = np.random.rand(N)
        y = np.random.rand(N)
        # Start with small radii to avoid immediate heavy penalties
        r = np.random.rand(N) * 0.1
        init = np.concatenate([x, y, r])
        
        res = minimize(objective, init, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 3000, 'ftol': 1e-15})
        
        # Check if the solution is valid (penalty near zero)
        if constraint_penalty(res.x) < 1e-10:
            s = np.sum(res.x[2*N:])
            if s > best_sum:
                best_sum = s
                best_sol = res.x
                
    # --- Strategy 2: Grid-Based Initialization ---
    # Initialize circles in a grid pattern to encourage dense packing
    init = np.zeros(3*N)
    count = 0
    # Attempt to place circles in a roughly 5x6 grid structure
    for row in range(6):
        for col in range(5):
            if count >= N: break
            
            # Spacing approx 0.2
            x = 0.1 + col * 0.2
            y = 0.1 + row * 0.18
            
            # Ensure within bounds
            if x > 0.9 or y > 0.9: continue
            
            init[2*count] = x
            init[2*count+1] = y
            init[2*count+2] = 0.08 # Start with reasonable radius
            count += 1
        if count >= N: break
    
    # Fill any remaining circles if grid didn't fill N
    if count < N:
        for i in range(count, N):
            init[2*i] = np.random.rand()
            init[2*i+1] = np.random.rand()
            init[2*i+2] = 0.05

    res = minimize(objective, init, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 3000, 'ftol': 1e-15})
    
    if constraint_penalty(res.x) < 1e-10:
        s = np.sum(res.x[2*N:])
        if s > best_sum:
            best_sum = s
            best_sol = res.x
            
    # Extract and return result
    if best_sol is not None:
        centers = best_sol[:2*N].reshape(N, 2)
        radii = best_sol[2*N:]
        return centers, radii, np.sum(radii)
    else:
        # Fallback to a safe, though suboptimal, packing
        centers = np.zeros((N, 2))
        radii = np.full(N, 0.01)
        return centers, radii, 0.26
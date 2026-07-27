import numpy as np
import math
import scipy.optimize
import random

def distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def check_validity(centers, radii):
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 0: return False
        if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(centers[i], centers[j])
            if dist < radii[i] + radii[j] - 1e-7:
                return False
    return True

def objective_and_gradient(params, n):
    """
    Objective: Maximize sum of radii.
    We minimize negative sum.
    Constraints are handled via penalty in the cost function or via bounds/constraints in optimizer.
    Here we use a penalty method within the objective for flexibility.
    """
    centers = params[0:2*n].reshape((n, 2))
    radii = params[2*n:]
    
    # Penalty parameters
    penalty_overlap = 1000.0
    penalty_boundary = 1000.0
    
    cost = -np.sum(radii) # We want to maximize sum, so minimize negative sum
    grad_centers = np.zeros((n, 2))
    grad_radii = np.zeros(n)
    
    # Boundary penalties
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left
        val = r - x
        if val > 0:
            cost += penalty_boundary * val**2
            grad_centers[i, 0] -= penalty_boundary * 2 * val
            grad_radii[i] += penalty_boundary * 2 * val
            
        # Right
        val = x - (1.0 - r)
        if val > 0:
            cost += penalty_boundary * val**2
            grad_centers[i, 0] -= penalty_boundary * 2 * val
            grad_radii[i] += penalty_boundary * 2 * val # r increases cost (pushes x out)
            
        # Bottom
        val = r - y
        if val > 0:
            cost += penalty_boundary * val**2
            grad_centers[i, 1] -= penalty_boundary * 2 * val
            grad_radii[i] += penalty_boundary * 2 * val
            
        # Top
        val = y - (1.0 - r)
        if val > 0:
            cost += penalty_boundary * val**2
            grad_centers[i, 1] -= penalty_boundary * 2 * val
            grad_radii[i] += penalty_boundary * 2 * val

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            r_sum = radii[i] + radii[j]
            
            if dist < 1e-9: dist = 1e-9 # Avoid division by zero
            
            overlap = r_sum - dist
            if overlap > 0:
                cost += penalty_overlap * overlap**2
                
                # Gradient w.r.t centers
                # d(overlap)/d(ci) = - d(dist)/d(ci)
                # d(dist)/d(ci) = (ci - cj) / dist
                # So term is - (ci - cj)/dist
                factor = penalty_overlap * 2 * overlap
                
                grad_centers[i, 0] -= factor * (dx / dist)
                grad_centers[i, 1] -= factor * (dy / dist)
                grad_centers[j, 0] += factor * (dx / dist)
                grad_centers[j, 1] += factor * (dy / dist)
                
                # Gradient w.r.t radii
                grad_radii[i] += factor * 1.0
                grad_radii[j] += factor * 1.0

    # Natural gradient for sum of radii
    grad_radii -= 1.0
    
    grad = np.concatenate([grad_centers.flatten(), grad_radii])
    return cost, grad

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Helper to initialize
    def init_grid():
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # 5x5 grid
        idx = 0
        step = 0.18 # Spacing
        start = 0.5 - 2*step
        
        # Create a slightly perturbed grid to avoid symmetry issues
        for r in range(5):
            for c in range(5):
                if idx < 25:
                    centers[idx] = [start + c*step + random.uniform(-0.01, 0.01), 
                                    start + r*step + random.uniform(-0.01, 0.01)]
                    radii[idx] = 0.09
                    idx += 1
        
        # 26th circle in center
        if idx < 26:
            centers[idx] = [0.5 + random.uniform(-0.05, 0.05), 0.5 + random.uniform(-0.05, 0.05)]
            radii[idx] = 0.02
            idx += 1
            
        return centers, radii

    def init_random():
        centers = np.random.rand(n, 2)
        radii = np.random.rand(n) * 0.05 + 0.02
        # Ensure inside
        for i in range(n):
            radii[i] = min(radii[i], 0.5)
            centers[i, 0] = max(radii[i], min(1-radii[i], centers[i, 0]))
            centers[i, 1] = max(radii[i], min(1-radii[i], centers[i, 1]))
        return centers, radii

    def optimize(centers, radii):
        # Flatten parameters
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Bounds
        # x, y in [0, 1]
        # r >= 0
        bounds = []
        for _ in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
        for _ in range(n):
            bounds.append((0, 0.5)) # r
            
        # Use L-BFGS-B or similar, but it doesn't support constraints directly in this form without penalty.
        # We implemented penalty in objective.
        
        # To make it work better, we can scale radii up during process?
        # No, let's just minimize.
        
        try:
            res = scipy.optimize.minimize(
                lambda p: objective_and_gradient(p, n)[0],
                x0,
                method='L-BFGS-B',
                jac=lambda p: objective_and_gradient(p, n)[1],
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            return res.x, -res.fun
        except Exception:
            return x0, -np.sum(radii)

    # Try multiple restarts
    for attempt in range(20):
        if attempt < 10:
            centers, radii = init_grid()
        else:
            centers, radii = init_random()
            
        # Pre-optimization: resolve overlaps aggressively
        for _ in range(100):
            for i in range(n):
                for j in range(i+1, n):
                    dist = distance(centers[i], centers[j])
                    r_sum = radii[i] + radii[j]
                    if dist < r_sum:
                        # Push apart
                        if dist < 1e-9:
                            centers[i][0] -= 0.01
                            centers[i][1] -= 0.01
                        else:
                            overlap = (r_sum - dist) / 2.0
                            dx = (centers[j][0] - centers[i][0]) / dist
                            dy = (centers[j][1] - centers[i][1]) / dist
                            centers[i][0] -= dx * overlap
                            centers[i][1] -= dy * overlap
                            centers[j][0] += dx * overlap
                            centers[j][1] += dy * overlap
            
            # Boundary push
            for i in range(n):
                r = radii[i]
                if centers[i][0] < r: centers[i][0] = r
                if centers[i][0] > 1-r: centers[i][0] = 1-r
                if centers[i][1] < r: centers[i][1] = r
                if centers[i][1] > 1-r: centers[i][1] = 1-r
        
        opt_params, current_sum = optimize(centers, radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = opt_params[0:2*n].reshape((n, 2))
            best_radii = opt_params[2*n:]

    # Final polish with SLSQP using constraints if possible? 
    # The penalty method is robust. Let's just ensure validity.
    
    # Check validity and print if invalid
    if not check_validity(best_centers, best_radii):
        print("Warning: Result not valid, trying to fix...")
        # Simple fix: shrink radii slightly
        factor = 1.0
        while not check_validity(best_centers, best_radii * factor):
            factor -= 0.01
            if factor < 0.5: break
    
    final_radii = best_radii * (1 if check_validity(best_centers, best_radii) else 0.99) # Safety
    if not check_validity(best_centers, final_radii):
         # Fallback to a safe packing
         # 5x5 grid of 0.09 + one small
         safe_centers = np.zeros((n, 2))
         safe_radii = np.ones(n) * 0.09
         idx = 0
         for r in range(5):
             for c in range(5):
                 if idx < 25:
                     safe_centers[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                     idx += 1
         safe_centers[25] = [0.5, 0.5]
         safe_radii[25] = 0.01
         best_centers = safe_centers
         final_radii = safe_radii
         best_sum = np.sum(final_radii)

    return best_centers, final_radii, np.sum(final_radii)

# Note: The run_packing function must be defined exactly as requested.
# The above logic is wrapped inside.
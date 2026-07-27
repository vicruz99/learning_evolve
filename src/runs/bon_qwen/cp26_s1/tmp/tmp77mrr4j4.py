import numpy as np
from scipy.optimize import differential_evolution
import math

def objective_function(params, n_circles=26):
    """
    Objective function to maximize sum of radii.
    Returns negative sum of radii (to be minimized) minus penalties for violations.
    
    Params structure: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    """
    # Penalty constants
    overlap_penalty = 100.0
    boundary_penalty = 100.0
    
    total_radius = 0.0
    penalty = 0.0
    
    # Parse centers and radii
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        idx = i * 3
        x = params[idx]
        y = params[idx+1]
        r = params[idx+2]
        
        centers[i] = [x, y]
        radii[i] = r
        
        # Check non-negative radius
        if r < 0:
            penalty += boundary_penalty * 100.0
            continue
            
        total_radius += r
        
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += boundary_penalty * ((x - r) if x - r < 0 else 0)
            penalty += boundary_penalty * ((x + r - 1) if x + r > 1 else 0)
            penalty += boundary_penalty * ((y - r) if y - r < 0 else 0)
            penalty += boundary_penalty * ((y + r - 1) if y + r > 1 else 0)

    # Check overlap constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = math.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += overlap_penalty * overlap

    # We want to maximize sum of radii, so we minimize negative sum
    return -total_radius + penalty

def run_packing():
    n_circles = 26
    bounds = []
    
    # Bounds for x, y in [0, 1], r in [0, 0.5] (radius cannot exceed 0.5)
    for _ in range(n_circles):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Initial guess: Hexagonal packing
    # We will run differential evolution which uses its own population
    # But providing a good seed via initial_guess might help, though DE doesn't take a single guess.
    # Instead, we rely on DE's global search capabilities.
    
    # To speed up and improve results, we can run multiple trials or use a sophisticated strategy.
    # Here we use a standard differential evolution with high population size.
    
    # Constraints can be tricky for DE. The penalty method is robust but can be slow if penalties are high.
    # Let's refine the penalty to be quadratic to encourage constraint satisfaction.
    
    def penalized_objective(params):
        total_radius = 0.0
        penalty = 0.0
        overlap_coeff = 1000.0 # High penalty for overlap
        boundary_coeff = 1000.0
        
        centers_x = params[0::3]
        centers_y = params[1::3]
        radii = params[2::3]
        
        # Check radii bounds roughly
        for r in radii:
            if r < 0:
                penalty += 1000.0 * abs(r)
                
        for i in range(n_circles):
            x, y, r = centers_x[i], centers_y[i], radii[i]
            total_radius += r
            
            # Boundary penalties
            if x - r < 0:
                penalty += boundary_coeff * (x - r)**2
            if x + r > 1:
                penalty += boundary_coeff * (x + r - 1)**2
            if y - r < 0:
                penalty += boundary_coeff * (y - r)**2
            if y + r > 1:
                penalty += boundary_coeff * (y + r - 1)**2
                
        # Overlap penalties
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_sq = (centers_x[i] - centers_x[j])**2 + (centers_y[i] - centers_y[j])**2
                dist = math.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += overlap_coeff * (min_dist - dist)**2
                    
        return -total_radius + penalty

    # Differential Evolution settings
    # strategy='best1bin', maxiter=1000, popsize=15 (default is 15*len(x0))
    # Since dimension is 26*3 = 78, popsize 15*78 is large but manageable.
    # To save time, we can reduce popsize or maxiter, but we need high accuracy.
    # Let's try a reasonable balance.
    
    try:
        result = differential_evolution(
            penalized_objective,
            bounds,
            seed=42,
            maxiter=1500,
            popsize=20,
            mutation=(0.5, 1.5),
            recombination=0.9,
            tol=1e-7,
            updating='deferred',
            polish=True
        )
        
        best_params = result.x
        
        # Extract best solution
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        
        for i in range(n_circles):
            idx = i * 3
            centers[i, 0] = best_params[idx]
            centers[i, 1] = best_params[idx+1]
            radii[i] = best_params[idx+2]
            
        sum_radii = np.sum(radii)
        
        return centers, radii, sum_radii
        
    except Exception as e:
        # Fallback to a simple grid packing if optimization fails
        centers_fallback = []
        radii_fallback = []
        r = 0.09 # Safe radius
        # 5x5 grid plus 1
        grid_points = [(0.1 + 0.2*i, 0.1 + 0.2*j) for i in range(5) for j in range(5)]
        for x, y in grid_points[:25]:
            centers_fallback.append([x, y])
            radii_fallback.append(r)
        # Add 26th circle in a gap
        centers_fallback.append([0.2, 0.2])
        radii_fallback.append(0.04)
        
        return np.array(centers_fallback), np.array(radii_fallback), sum(radii_fallback)

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Min radius: {np.min(r)}, Max radius: {np.max(r)}")
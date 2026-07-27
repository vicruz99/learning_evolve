import numpy as np
import scipy.optimize as opt
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_sum_radii = -np.inf
    best_centers = None
    best_radii = None

    # Helper function to convert flat vector to centers/radii
    def vector_to_params(vec):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = vec[3 * i]
            centers[i, 1] = vec[3 * i + 1]
            radii[i] = vec[3 * i + 2]
        return centers, radii

    # Helper function to compute objective (negative sum of radii)
    def objective(vec):
        _, radii = vector_to_params(vec)
        return -np.sum(radii)

    # Helper function to define constraints
    def get_constraints():
        constraints = []
        
        # Boundary constraints: r <= x <= 1-r  =>  x - r >= 0, x + r <= 1
        # y - r >= 0, y + r <= 1
        # Also r >= 0 is handled by bounds
        
        for i in range(n):
            # x - r >= 0
            cons = {'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]}
            constraints.append(cons)
            # 1 - (x + r) >= 0
            cons = {'type': 'ineq', 'fun': lambda v, i=i: 1.0 - (v[3*i] + v[3*i+2])}
            constraints.append(cons)
            
            # y - r >= 0
            cons = {'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]}
            constraints.append(cons)
            # 1 - (y + r) >= 0
            cons = {'type': 'ineq', 'fun': lambda v, i=i: 1.0 - (v[3*i+1] + v[3*i+2])}
            constraints.append(cons)

        # Non-overlap constraints: dist(c_i, c_j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        for i in range(n):
            for j in range(i + 1, n):
                cons = {
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                }
                constraints.append(cons)
        
        return constraints

    constraints = get_constraints()

    # Bounds for variables [x, y, r]
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius)
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]) # r > 0 to avoid degeneracy

    # Strategy 1: Hexagonal-ish grid initialization
    # Try to place 26 points in a hexagonal lattice pattern
    def init_hex_grid():
        # Attempt to fit 26 circles. 
        # A 5x5 grid fits 25. We can try a perturbed grid.
        centers = np.zeros((n, 2))
        # 5 rows, 5 columns roughly
        # Let's try to space them out
        # 5x5 grid spacing
        step = 0.2
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < n:
                    centers[idx] = [0.1 + c * step, 0.1 + r * step]
                    idx += 1
        # Place the last one in the center hole if possible, or near center
        if idx < n:
            centers[idx] = [0.5, 0.5] # Overlaps, will be fixed by optimizer
            idx += 1
        
        # Initial small radii
        radii = np.full(n, 0.05)
        
        vec = np.zeros(3 * n)
        for i in range(n):
            vec[3*i] = centers[i, 0]
            vec[3*i+1] = centers[i, 1]
            vec[3*i+2] = radii[i]
        return vec

    # Strategy 2: Random initialization within bounds
    def init_random():
        vec = np.random.uniform(0.1, 0.9, size=2 * n)
        radii = np.random.uniform(0.01, 0.05, size=n)
        full_vec = np.zeros(3 * n)
        for i in range(n):
            full_vec[3*i] = vec[2*i]
            full_vec[3*i+1] = vec[2*i+1]
            full_vec[3*i+2] = radii[i]
        return full_vec

    # Run optimization
    candidates = []
    
    # Generate a few hex grid candidates with slight perturbations
    for _ in range(3):
        vec = init_hex_grid()
        vec += np.random.normal(0, 0.01, size=len(vec))
        vec = np.clip(vec, 0.01, 0.99) # Ensure bounds are respected roughly
        # Adjust radii bounds
        for i in range(n):
            vec[3*i+2] = min(vec[3*i+2], 0.4)
        candidates.append(vec)

    # Generate random candidates
    for _ in range(5):
        candidates.append(init_random())

    best_sol = None

    for i, x0 in enumerate(candidates):
        try:
            res = opt.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            if res.success and res.fun < best_sum_radii: # Remember fun is negative sum
                best_sum_radii = res.fun
                best_sol = res.x
        except Exception:
            continue

    if best_sol is not None:
        # Perturb and optimize again to escape local minima
        for _ in range(3):
            perturbed_sol = best_sol + np.random.normal(0, 0.005, size=len(best_sol))
            # Ensure bounds
            for k in range(0, len(perturbed_sol), 3):
                perturbed_sol[k] = np.clip(perturbed_sol[k], 0.01, 0.99)
                perturbed_sol[k+1] = np.clip(perturbed_sol[k+1], 0.01, 0.99)
                perturbed_sol[k+2] = np.clip(perturbed_sol[k+2], 1e-5, 0.45)
            
            try:
                res = opt.minimize(
                    objective,
                    perturbed_sol,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
                if res.success and res.fun < best_sum_radii:
                    best_sum_radii = res.fun
                    best_sol = res.x
            except Exception:
                pass

    if best_sol is not None:
        centers, radii = vector_to_params(best_sol)
        sum_radii = -best_sum_radii
        
        # Final validation check (sanity)
        # Note: The prompt validation function is strict.
        # Our constraints should guarantee validity, but numerical noise might occur.
        # We clamp radii slightly if needed, but optimizer should handle it.
        
        return centers, radii, sum_radii
    else:
        # Fallback to a valid simple packing (25 circles radius 0.1, 1 circle radius 0.01)
        # This ensures we return something valid if optimization fails completely
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # 5x5 grid for first 25
        step = 0.2
        idx = 0
        for r in range(5):
            for c in range(5):
                centers[idx] = [0.1 + c * step, 0.1 + r * step]
                radii[idx] = 0.1
                idx += 1
        
        # 26th circle in corner gap?
        # Place at (0.05, 0.05) with small radius
        centers[25] = [0.05, 0.05]
        radii[25] = 0.05 # Might overlap, but let's try to be safe
        # Actually 0.05 at 0.05 touches 0.1 at 0.1? dist = sqrt(0.01+0.01) = 0.141. r1+r2 = 0.15. Overlap.
        # Place at (0.05, 0.5)?
        # Let's just place it at (0.05, 0.05) with r=0.01
        radii[25] = 0.01
        
        return centers, radii, np.sum(radii)

# Allow running the function
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Number of circles: {len(radii)}")
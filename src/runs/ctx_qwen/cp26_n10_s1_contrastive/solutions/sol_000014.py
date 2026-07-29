# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 002dac80) state=e93fa8a8 sum of radii=2.175783 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
import math

def get_optimal_radii(centers):
    """
    Solves the linear program to find radii that maximize sum of radii
    subject to non-overlap and boundary constraints for fixed centers.
    """
    n = centers.shape[0]
    
    # Bounds for each radius: 0 <= r_i <= distance to nearest wall
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        if max_r < 0: max_r = 0
        bounds.append((0, max_r))
    
    # Inequality constraints: r_i + r_j <= distance(i, j)
    # There are n*(n-1)/2 such constraints
    num_constraints = n * (n - 1) // 2
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((centers[i,0] - centers[j,0])**2 + (centers[i,1] - centers[j,1])**2)
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    # Objective: Maximize sum(r) => Minimize -sum(r)
    c = np.full(n, -1.0)
    
    try:
        # Use 'highs' method if available, else fallback
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback to a safe small radius if LP fails
            return np.full(n, 0.001)
    except Exception:
        return np.full(n, 0.001)

def calculate_score(centers):
    radii = get_optimal_radii(centers)
    return np.sum(radii), radii

def run_packing():
    n = 26
    # Initialize centers with a hexagonal-like grid perturbed
    # We want a dense initial configuration
    # Try to place 26 circles in a way that respects boundaries roughly
    # A 5x5 grid has 25 spots. We add 1.
    # Let's use a randomized initialization with multiple restarts to find global optimum.
    
    best_score = -1
    best_centers = None
    best_radii = None
    
    # Number of random restarts
    n_restarts = 20
    
    for _ in range(n_restarts):
        # Random initialization
        # Place centers in [0.1, 0.9] to ensure some initial room
        centers = np.random.uniform(0.05, 0.95, (n, 2))
        
        # Local optimization using Hill Climbing / Simulated Annealing
        # We will perturb centers and accept if score improves
        
        current_score, current_radii = calculate_score(centers)
        
        # Learning rate for perturbation
        step_size = 0.05
        
        for iteration in range(500):
            # Randomly perturb one or more centers
            # Perturb all centers slightly
            perturbations = np.random.normal(0, step_size, centers.shape)
            new_centers = centers + perturbations
            
            # Clip to valid region [0, 1]
            # Actually centers can be anywhere, but radii will be 0 if outside.
            # Better to keep centers in [0,1] to avoid trivial solutions?
            # No, centers must be in [0,1] for circles to be inside?
            # Wait, if center is at -0.1 and radius 0, it's valid?
            # But radius must be non-negative.
            # The LP handles boundaries. If center is outside, max_r is 0 (or negative clamped to 0).
            # So valid centers are preferred.
            
            # Clip to [0,1] to keep search space relevant
            new_centers = np.clip(new_centers, 0, 1)
            
            new_score, new_radii = calculate_score(new_centers)
            
            # Accept if better
            if new_score > current_score:
                centers = new_centers
                current_score = new_score
                current_radii = new_radii
                # Reduce step size slowly
                step_size *= 0.995
            else:
                # Simulated annealing: accept worse with probability
                # Temperature decreases over iterations
                temp = 0.01 * (1 - iteration / 500)
                if temp > 0 and np.random.rand() < math.exp((new_score - current_score) / max(temp, 1e-9)):
                    centers = new_centers
                    current_score = new_score
                    current_radii = new_radii
                    step_size *= 0.99
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers
            best_radii = current_radii

    # Final refinement with smaller step size
    centers = best_centers
    current_score, current_radii = calculate_score(centers)
    step_size = 0.01
    for iteration in range(1000):
        perturbations = np.random.normal(0, step_size, centers.shape)
        new_centers = np.clip(centers + perturbations, 0, 1)
        new_score, new_radii = calculate_score(new_centers)
        
        if new_score > current_score:
            centers = new_centers
            current_score = new_score
            current_radii = new_radii
            step_size *= 0.998
    
    # Ensure radii are non-negative and valid
    final_radii = get_optimal_radii(centers)
    
    # Double check validity (though LP should ensure it)
    # Just in case of numerical issues in LP, clamp radii
    # But LP is strict.
    
    return centers, final_radii, np.sum(final_radii)

if __name__ == "__main__":
    # Run a quick test
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Validate
    # Note: validate_packing is provided in prompt, not here.

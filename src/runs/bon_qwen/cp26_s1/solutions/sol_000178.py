# sol_000178 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b137705a) state=086d048d sum of radii=2.210060 correctness=1.0
# stdout(first 200): Phase 1: Initializing with force-based expansion... Phase 1 result: Sum of radii = 1.30000 Phase 2: Optimizing centers and radii...
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

# Constants
N_CIRCLES = 26
SQ_SIDE = 1.0

def compute_overlap_penalty(centers, radii):
    """
    Computes the penalty for overlaps between circles and boundary violations.
    Returns a scalar penalty value.
    """
    penalty = 0.0
    n = centers.shape[0]
    
    # Boundary penalties: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    # Equivalent to: r - x <= 0, x + r - 1 <= 0, etc.
    # We penalize positive violations squared.
    
    # Vectorized boundary checks
    # x - r < 0 => r - x > 0
    pen_x_min = radii - centers[:, 0]
    pen_x_min = np.where(pen_x_min > 0, pen_x_min, 0)
    
    # x + r > 1 => x + r - 1 > 0
    pen_x_max = centers[:, 0] + radii - 1.0
    pen_x_max = np.where(pen_x_max > 0, pen_x_max, 0)
    
    # y - r < 0 => r - y > 0
    pen_y_min = radii - centers[:, 1]
    pen_y_min = np.where(pen_y_min > 0, pen_y_min, 0)
    
    # y + r > 1 => y + r - 1 > 0
    pen_y_max = centers[:, 1] + radii - 1.0
    pen_y_max = np.where(pen_y_max > 0, pen_y_max, 0)
    
    boundary_penalty = np.sum(pen_x_min**2) + np.sum(pen_x_max**2) + \
                       np.sum(pen_y_min**2) + np.sum(pen_y_max**2)
    
    # Pairwise overlap penalties
    # We iterate i < j
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap**2
                
    penalty += boundary_penalty
    return penalty

def objective_function(vars_flat, weights):
    """
    Objective function to minimize.
    We want to maximize sum(radii), so we minimize -sum(radii).
    We add penalty terms for overlaps and boundary violations.
    
    vars_flat: array of shape [x1, y1, r1, x2, y2, r2, ...]
    """
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        idx = 3 * i
        centers[i, 0] = vars_flat[idx]
        centers[i, 1] = vars_flat[idx + 1]
        radii[i] = vars_flat[idx + 2]
        
    # Penalty term
    penalty_val = compute_overlap_penalty(centers, radii)
    
    # Objective: -sum(radii) + penalty_weight * penalty
    obj_val = -np.sum(radii) + weights['penalty_w'] * penalty_val
    
    return obj_val

def force_based_expansion():
    """
    Initializes a valid packing of 26 circles with equal radii using force-based repulsion.
    This serves as a warm start for the optimization.
    """
    np.random.seed(42) # For reproducibility
    
    # Initial placement: random grid-like or random
    centers = np.random.rand(N_CIRCLES, 2) * 0.8 + 0.1 # Keep away from edges initially
    r = 0.05 # Initial radius
    
    # Simulation parameters
    dt = 0.05
    repulsion_strength = 10.0
    damping = 0.9
    
    # Number of expansion steps
    total_steps = 2000
    radius_growth_rate = 0.0005
    
    velocities = np.zeros((N_CIRCLES, 2))
    
    best_sum_radii = 0
    best_centers = centers.copy()
    best_radii = np.full(N_CIRCLES, r)
    
    for step in range(total_steps):
        # Increase radius slightly
        if step % 10 == 0:
            r += radius_growth_rate
        
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Repulsion between circles
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                
                # Overlap distance
                overlap = 2 * r - dist
                
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    if dist > 1e-9:
                        fx = (dx / dist) * overlap * repulsion_strength
                        fy = (dy / dist) * overlap * repulsion_strength
                    else:
                        # Random push if centers coincide
                        fx = (np.random.rand() - 0.5) * 0.1
                        fy = (np.random.rand() - 0.5) * 0.1
                    
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
        
        # Boundary forces
        for i in range(N_CIRCLES):
            # Left wall
            if centers[i, 0] - r < 0:
                forces[i, 0] += (r - centers[i, 0]) * repulsion_strength
            # Right wall
            if centers[i, 0] + r > 1.0:
                forces[i, 0] -= (centers[i, 0] + r - 1.0) * repulsion_strength
            # Bottom wall
            if centers[i, 1] - r < 0:
                forces[i, 1] += (r - centers[i, 1]) * repulsion_strength
            # Top wall
            if centers[i, 1] + r > 1.0:
                forces[i, 1] -= (centers[i, 1] + r - 1.0) * repulsion_strength
                
        # Update velocities and positions
        velocities += forces * dt
        velocities *= damping # Damping to stabilize
        centers += velocities * dt
        
        # Clamp positions to [r, 1-r] strictly to avoid huge penalties
        centers[:, 0] = np.clip(centers[:, 0], r, 1.0 - r)
        centers[:, 1] = np.clip(centers[:, 1], r, 1.0 - r)
        
        # Check validity and save best
        # We do a rough check every 100 steps to avoid overhead
        if step % 100 == 0:
            # Reconstruct radii array
            current_radii = np.full(N_CIRCLES, r)
            # Check for overlaps roughly
            valid = True
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 2*r - 1e-7:
                        valid = False
                        break
                if not valid: break
            
            # Check boundaries
            for i in range(N_CIRCLES):
                if centers[i, 0] < r - 1e-7 or centers[i, 0] > 1-r + 1e-7 or \
                   centers[i, 1] < r - 1e-7 or centers[i, 1] > 1-r + 1e-7:
                    valid = False
                    break
            
            if valid:
                current_sum = np.sum(current_radii)
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = centers.copy()
                    best_radii = current_radii.copy()
                    
    return best_centers, best_radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Main function to pack 26 circles.
    Returns (centers, radii, sum_radii).
    """
    # Phase 1: Get a good initial packing with equal radii
    print("Phase 1: Initializing with force-based expansion...")
    centers_init, radii_init = force_based_expansion()
    print(f"Phase 1 result: Sum of radii = {np.sum(radii_init):.5f}")
    
    # Phase 2: Refine using scipy.optimize with penalty method
    # We allow radii to vary to potentially increase the sum.
    
    # Flatten initial state
    # Format: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N_CIRCLES)
    for i in range(N_CIRCLES):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Penalty weights
    # Start with high weight to enforce constraints, but might need adjustment
    # Since we start valid, we can use a weight that pushes boundaries.
    weights = {'penalty_w': 1000.0}
    
    print("Phase 2: Optimizing centers and radii...")
    
    # Run optimization
    # L-BFGS-B is good for bound-constrained problems
    result = minimize(
        objective_function, 
        x0, 
        args=(weights,), 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    # Extract result
    x_opt = result.x
    centers_opt = np.zeros((N_CIRCLES, 2))
    radii_opt = np.zeros(N_CIRCLES)
    
    for i in range(N_CIRCLES):
        centers_opt[i, 0] = x_opt[3*i]
        centers_opt[i, 1] = x_opt[3*i+1]
        radii_opt[i] = x_opt[3*i+2]
        
    # If optimization resulted in slight violations due to numerical error or penalty balance,
    # we might need to shrink radii slightly to ensure strict validity.
    # However, with high penalty weight and starting from valid, it should be close.
    # Let's check and fix.
    
    # Safety clamp: ensure centers are within valid range for their radii
    for i in range(N_CIRCLES):
        r = radii_opt[i]
        centers_opt[i, 0] = np.clip(centers_opt[i, 0], r, 1.0 - r)
        centers_opt[i, 1] = np.clip(centers_opt[i, 1], r, 1.0 - r)
        
    # Check for overlaps and shrink if necessary
    # This is a greedy fix: if two circles overlap, reduce the larger radius or both
    # But since we want to maximize sum, shrinking is bad.
    # However, if the optimizer failed to satisfy constraints, we must.
    
    # Let's run a quick validation check logic here to adjust if needed
    # If we detect overlap, we reduce radii slightly.
    
    # Heuristic shrink if penalty is non-zero
    penalty = compute_overlap_penalty(centers_opt, radii_opt)
    if penalty > 1e-6:
        # Simple scaling down
        scale = 0.95
        radii_opt *= scale
        # Re-center? Or just clip. Clipping is safer.
        for i in range(N_CIRCLES):
            r = radii_opt[i]
            centers_opt[i, 0] = np.clip(centers_opt[i, 0], r, 1.0 - r)
            centers_opt[i, 1] = np.clip(centers_opt[i, 1], r, 1.0 - r)
            
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii

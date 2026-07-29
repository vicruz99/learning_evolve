# sol_000179 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8fc41863) state=2d51dcbf sum of radii=1.823350 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses a custom optimization starting from a perturbed grid.
    """
    n = 26
    
    # Initial guess: 5x5 grid with slight random perturbation to break symmetry
    # and a 26th circle squeezed into a gap.
    # We'll use 25 circles of radius ~0.095 and 1 small one.
    
    # Generate 5x5 grid
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    
    # Remove the center one (25th) to make space, or just add 26th?
    # 5x5 has 25. We need 26.
    # Let's keep 25 in grid, add 26th in center.
    # But center is occupied.
    # Let's just randomize positions slightly to allow optimizer to find a better layout.
    
    np.random.seed(42)
    centers = np.array(pts)
    # Add 26th point near a gap (e.g., 0.2, 0.2)
    centers = np.vstack([centers, [0.2, 0.2]])
    
    # Initial radii: all small to avoid overlap
    radii = np.full(n, 0.04)
    
    # Optimization Objective: Maximize sum of radii
    # We can parameterize the problem by fixing relative positions or optimizing centers and radii.
    # Since it's non-convex, we use a simple iterative repulsion-based expansion.
    
    # Better approach: Use scipy minimize with constraints? 
    # Constraints are non-linear. 
    # Let's use a "simulated annealing" style or "repulsion force" simulation.
    
    # Force-based simulation
    # Each circle pushes others away and tries to expand against walls.
    
    # Set bounds for centers
    low_bound = radii
    high_bound = 1 - radii
    
    def get_energy(centers, radii):
        # Negative of sum of radii
        # Plus penalty for overlap and boundary violation
        energy = -np.sum(radii)
        penalty = 0.0
        
        # Boundary penalties
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            # Soft constraints
            if x < r: penalty += (r - x)**2
            if x > 1-r: penalty += (x - (1-r))**2
            if y < r: penalty += (r - y)**2
            if y > 1-r: penalty += (y - (1-r))**2
            
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
                    
        return energy + penalty * 100.0 # Weight for penalty

    # We will optimize positions for a given set of radii, then increase radii.
    # Or optimize both.
    
    # Let's use a simplified "expanding" simulation
    # 1. Fix radii to a target sum.
    # 2. Optimize centers to minimize overlaps.
    # 3. If valid, increase radii.
    
    current_radii = np.full(n, 0.05) # Start small
    current_centers = np.copy(centers)
    
    # Shuffle centers to randomize
    np.random.shuffle(current_centers)
    
    best_sum = 0.0
    best_centers = current_centers
    best_radii = current_radii
    
    # Iterative expansion
    for step in range(200):
        # Try to increase radii slightly
        target_r = current_radii * 1.01
        
        # Optimize centers to fit these radii
        # We use a simple gradient descent on the overlap function
        centers_opt = np.copy(current_centers)
        
        for _ in range(50): # Inner optimization steps
            # Calculate repulsion forces
            forces = np.zeros_like(centers_opt)
            
            # Boundary forces (push inward if touching)
            for i in range(n):
                r = target_r[i]
                x, y = centers_opt[i]
                # Left wall
                if x - r < 0:
                    forces[i, 0] += (r - x) * 10
                # Right wall
                if x + r > 1:
                    forces[i, 0] -= (x + r - 1) * 10
                # Bottom wall
                if y - r < 0:
                    forces[i, 1] += (r - y) * 10
                # Top wall
                if y + r > 1:
                    forces[i, 1] -= (y + r - 1) * 10
            
            # Circle-circle repulsion
            for i in range(n):
                for j in range(i+1, n):
                    vec = centers_opt[i] - centers_opt[j]
                    dist = np.linalg.norm(vec)
                    if dist == 0:
                        dist = 1e-9
                        vec = np.random.rand(2) * 0.01 # Break symmetry
                    
                    min_dist = target_r[i] + target_r[j]
                    if dist < min_dist:
                        # Repulsion force proportional to overlap
                        repulsion = (min_dist - dist) / dist # normalized force
                        # Apply force to move apart
                        forces[i] += vec * repulsion * 5.0
                        forces[j] -= vec * repulsion * 5.0
            
            # Update positions
            step_size = 0.005
            centers_opt += forces * step_size
            
            # Clamp to bounds [0, 1]
            # Actually, we want centers to stay within [r, 1-r] ideally, 
            # but forces handle it. Just clamp to [0, 1] for safety.
            centers_opt = np.clip(centers_opt, 0, 1)
            
            # Also ensure radii constraints are roughly met by center clamping
            for i in range(n):
                r = target_r[i]
                centers_opt[i, 0] = np.clip(centers_opt[i, 0], r, 1-r)
                centers_opt[i, 1] = np.clip(centers_opt[i, 1], r, 1-r)
        
        current_centers = centers_opt
        current_radii = target_r
        
        # Check if valid (loose check)
        valid = True
        for i in range(n):
            r = current_radii[i]
            x, y = current_centers[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                valid = False
                break
            for j in range(i+1, n):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                if dist < current_radii[i] + current_radii[j] - 1e-6:
                    valid = False
                    break
        
        if valid:
            s = np.sum(current_radii)
            if s > best_sum:
                best_sum = s
                best_centers = np.copy(current_centers)
                best_radii = np.copy(current_radii)
        else:
            # If invalid, revert radii increase and maybe decrease slightly
            current_radii = best_radii
            current_centers = best_centers
            # Break if we can't grow further (convergence)
            # To be safe, just continue but don't update best.
            # But we might be stuck. Let's restart radii from best and perturb.
            # For this simple script, we'll just stop growing if it fails consistently.
            # Actually, let's just keep the best found so far.
            pass

    # Final polish: Run a few more optimization steps with the best radii to clean up positions
    # We want to ensure strict non-overlap.
    # Use a repulsion pass with the final radii.
    
    final_radii = best_radii
    final_centers = best_centers
    
    # Repulsion pass to ensure no overlaps
    for _ in range(100):
        forces = np.zeros_like(final_centers)
        for i in range(n):
            r = final_radii[i]
            x, y = final_centers[i]
            if x - r < 1e-7: forces[i, 0] += (r - x + 1e-7) * 10
            if x + r > 1 - 1e-7: forces[i, 0] -= (x + r - 1 + 1e-7) * 10
            if y - r < 1e-7: forces[i, 1] += (r - y + 1e-7) * 10
            if y + r > 1 - 1e-7: forces[i, 1] -= (y + r - 1 + 1e-7) * 10
            
            for j in range(i+1, n):
                vec = final_centers[i] - final_centers[j]
                dist = np.linalg.norm(vec)
                if dist == 0:
                    vec = np.random.rand(2) * 0.001
                    dist = 1e-9
                min_dist = final_radii[i] + final_radii[j]
                if dist < min_dist:
                    repulsion = (min_dist - dist)
                    forces[i] += vec * repulsion * 10.0
                    forces[j] -= vec * repulsion * 10.0
        
        final_centers += forces * 0.001
        # Clamp
        for i in range(n):
            r = final_radii[i]
            final_centers[i, 0] = np.clip(final_centers[i, 0], r, 1-r)
            final_centers[i, 1] = np.clip(final_centers[i, 1], r, 1-r)

    # Verify and return
    # Recalculate sum
    final_sum = np.sum(final_radii)
    
    return final_centers, final_radii, final_sum

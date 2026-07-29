# sol_000297 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fc92aa36) state=09875f70 sum of radii=1.735911 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Returns (centers, radii, sum_radii) for a packing of 26 circles in a unit square.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialization
    # We start with a perturbed grid. A 6x5 grid is dense but requires small radii.
    # We want to find a configuration that allows larger radii.
    # Let's try a 6x5 grid layout for centers initially, with small radii.
    cols = 6
    rows = 5
    # Spacing to fit 6 items in [0, 1] with some margin
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    centers = np.zeros((n, 2))
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count < n:
                centers[count, 0] = (c + 1) * spacing_x
                centers[count, 1] = (r + 1) * spacing_y
                count += 1
    
    # Initial radii: small enough to not overlap
    radii = np.full(n, 0.01)
    
    # 2. Optimization using a custom iterative repulsion/expansion method
    # This is often more robust for circle packing than generic gradient descent 
    # due to the non-smooth nature of the constraints (max function).
    
    # Parameters
    max_iter = 2000
    dt = 0.01 # Step size for center movement
    expansion_rate = 1.05 # Factor to increase radii each step (slowly decaying)
    repulsion_strength = 10.0
    
    # Warm up: Expand radii until contacts form
    # We perform a simple "grow circles" phase
    for step in range(max_iter):
        # Decay expansion rate to allow convergence
        current_expansion = 1.0 + 0.01 * math.exp(-step / 500)
        
        # Decay step size
        current_dt = dt * (1.0 / (1.0 + step/100))

        # --- Phase 1: Compute Forces and Update Radii ---
        forces = np.zeros_like(centers)
        
        # Check boundaries and neighbors
        # We calculate the 'slack' for each circle: how much it can grow
        # But we also need to move centers if they are trapped.
        
        # First, update radii based on current constraints
        # r_i <= dist(center_i, boundary)
        # r_i + r_j <= dist(center_i, center_j)
        
        # A simple way to maximize sum of radii is to set r_i to the limiting constraint
        # but this causes oscillations. Instead, we grow them slightly.
        
        # Calculate max possible radius for each circle given current positions
        # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # r_i <= dist(i,j) - r_j
        
        # We can solve this as a linear system or just iteratively update.
        # Let's do a single pass update:
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Boundary constraints
            max_r_bound = min(x, 1 - x, y, 1 - y)
            
            # Neighbor constraints
            # r_i <= d_ij - r_j  =>  r_i + r_j <= d_ij
            # We want to find max r_i such that r_i + r_j <= d_ij for all j
            # This is r_i <= d_ij - r_j
            # So max_r_neighbor = min_j (d_ij - r_j)
            
            # However, we don't want to force r_i to exactly the limit, 
            # as that might block expansion of others.
            # Instead, we just ensure validity and grow.
            
            # Calculate current limiting radius from neighbors
            limit_from_neighbors = np.inf
            for j in range(n):
                if i == j: continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.hypot(dx, dy)
                # Constraint: r_i + r_j <= dist
                # r_i <= dist - r_j
                limit = dist - radii[j]
                if limit < limit_from_neighbors:
                    limit_from_neighbors = limit
            
            max_r = min(max_r_bound, limit_from_neighbors)
            
            # Ensure non-negative
            if max_r < 0: max_r = 0
            
            # Grow radius
            # We try to set r_i to a value between current r and max_r
            # But to maximize sum, we want r_i as large as possible.
            # However, increasing r_i restricts neighbors.
            # A balanced approach: r_i_new = r_i + alpha * (max_r - r_i)
            alpha = 0.5 # Aggressiveness of growth
            radii[i] = r + alpha * (max_r - r)
            
            # If radius shrunk, we might need to move center to free space?
            # Or just let the repulsion handle it in next step.
            # Actually, if max_r < r, we must shrink.
            if radii[i] > max_r:
                radii[i] = max_r

        # --- Phase 2: Repulsion (Move Centers) ---
        # If circles are touching/overlapping, move them apart.
        # We want to move them into "gaps" to allow future expansion.
        # Force is proportional to overlap.
        
        for i in range(n):
            fx, fy = 0, 0
            
            # Boundary repulsion: Push away from walls if touching
            x, y = centers[i]
            r = radii[i]
            
            # Walls
            if x - r < 1e-9: fx += repulsion_strength * (r - x)
            if x + r > 1 - 1e-9: fx -= repulsion_strength * (x + r - 1)
            if y - r < 1e-9: fy += repulsion_strength * (r - y)
            if y + r > 1 - 1e-9: fy -= repulsion_strength * (y + r - 1)
            
            # Neighbor repulsion
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.hypot(dx, dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap amount
                    overlap = min_dist - dist
                    # Force vector (normalized) * overlap
                    # Push i away from j
                    fx += repulsion_strength * overlap * (dx / dist)
                    fy += repulsion_strength * overlap * (dy / dist)
            
            forces[i, 0] = fx
            forces[i, 1] = fy

        # Apply forces
        # Cap velocity/step to prevent explosion
        max_force = np.linalg.norm(forces)
        if max_force > 0:
            # Normalize forces to unit vector, scale by step
            # Or just apply force directly with damping
            centers += forces * current_dt
            
            # Keep centers within bounds (hard clamp)
            centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1 - 1e-9)
            centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1 - 1e-9)
            
        # Optional: Small random perturbation to escape local minima
        if step % 100 == 0:
            centers += np.random.normal(0, 0.005, size=centers.shape)
            centers[:, 0] = np.clip(centers[:, 0], 1e-9, 1 - 1e-9)
            centers[:, 1] = np.clip(centers[:, 1], 1e-9, 1 - 1e-9)

    # 3. Final Validation and Cleanup
    # Ensure no overlaps and boundaries are respected
    # Since numerical errors might exist, we clamp radii slightly if needed
    # But the loop should have stabilized.
    
    # Check for any remaining tiny overlaps and shrink radii if necessary
    # This is a safety measure
    for i in range(n):
        # Boundary
        x, y = centers[i]
        r = radii[i]
        r = min(r, x, 1-x, y, 1-y)
        
        # Neighbors
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.hypot(dx, dy)
            # r_i + r_j <= dist
            # We need to reduce radii if sum > dist
            # To maximize sum, we should reduce them equally? Or just clamp?
            # If r_i + r_j > dist, we have overlap.
            # Let's reduce the sum of radii to dist.
            # Distribute reduction? 
            # Simple clamp: r_i = min(r_i, dist - r_j)
            # But this might be asymmetric.
            # A better way for validation is just to ensure valid.
            pass
        
        radii[i] = r # Apply boundary clamp

    # Re-check neighbor constraints and adjust if strictly violated
    # Iterate a few times to resolve overlaps by shrinking
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.hypot(dx, dy)
                if dist < radii[i] + radii[j]:
                    overlap = radii[i] + radii[j] - dist
                    # Reduce both by half overlap
                    radii[i] -= overlap / 2
                    radii[j] -= overlap / 2
                    changed = True
        if not changed: break
    
    # Ensure non-negative radii
    radii = np.maximum(radii, 0)
    
    # Final boundary check
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper to validate (as requested in prompt, though we just return values)
# The prompt says "We will run the below validation function".
# We just need to implement run_packing.

if __name__ == "__main__":
    import numpy as np
    # Re-define validate here for local testing if needed, 
    # but the system will use the provided one.
    
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any(): return False
        if np.isnan(radii).any(): return False
        for i in range(n):
            if radii[i] < 0: return False
            elif np.isnan(radii[i]): return False
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12:
                    return False
        return True

    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")
    print(f"Radii: {r}")

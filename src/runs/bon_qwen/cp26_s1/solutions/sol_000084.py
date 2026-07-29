# sol_000084 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b0810f40) state=c1722d08 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    np.random.seed(42)
    N = 26
    
    # 1. Initialization: Hexagonal Grid
    # We try to fit a hexagonal pattern. 
    # Rows: 5, 6, 5, 6, 4 sums to 26.
    # Vertical spacing h = sqrt(3)/2 * diameter? No, for radius r, spacing is r*sqrt(3).
    # Let's start with a reasonable radius estimate. 
    # For N=26, r ~ 0.1.
    
    r_est = 0.1
    centers = []
    radii = []
    
    # Row configuration: 5, 6, 5, 6, 4
    row_counts = [5, 6, 5, 6, 4]
    
    # Vertical positions for rows
    # Height of square = 1. 5 rows.
    # Vertical spacing between row centers = r_est * sqrt(3)
    # Total height spanned by centers = 4 * r_est * sqrt(3)
    # We need to fit margins.
    
    dy = r_est * math.sqrt(3)
    # Center the rows vertically
    y_offset = (1.0 - 4 * dy) / 2.0
    
    current_circle_idx = 0
    
    for i, count in enumerate(row_counts):
        # Horizontal spacing = 2 * r_est
        dx = 2 * r_est
        # Total width of row = (count - 1) * dx
        row_width = (count - 1) * dx
        # Center the row horizontally
        x_offset = (1.0 - row_width) / 2.0
        
        # Stagger odd/even rows?
        # In hexagonal packing, rows are usually shifted by dx/2 = r_est
        # But since we have varying counts, we align them to fit best.
        # Let's shift rows with even index (0, 2, 4) or (1, 3)?
        # Usually rows with more circles are the "main" ones.
        # Let's shift rows with count 5 (indices 0, 2) or 6?
        # Let's try shifting rows with count 5 by r_est to the right relative to count 6?
        # Actually, standard hex: row k and k+1 are shifted by r.
        
        shift = 0
        if i % 2 == 1: # Shift rows 1 and 3 (counts 6)
             shift = r_est # Shift right by r
             # But we must ensure we stay in bounds. 
             # If we shift right, x_start increases.
             # Let's recalculate offset considering shift?
             # Actually, if we shift, the width constraint is tighter.
             # Let's just place them and let optimization fix it.
             pass 
        
        # For now, simple centered placement without shift for initialization
        # Optimization will stagger them.
        for j in range(count):
            x = x_offset + j * dx
            y = y_offset + i * dy
            centers.append([x, y])
            radii.append(r_est)
            
            current_circle_idx += 1
            
    centers = np.array(centers)
    radii = np.array(radii)
    
    # 2. Force-Directed Optimization
    # We want to maximize sum(r_i) subject to non-overlap and boundaries.
    # We simulate a system where circles repel, and a "pressure" expands them.
    
    # Parameters
    n_iter = 3000
    pressure_coeff = 0.001 # How much radius increases per step if no overlap
    repulsion_coeff = 10.0 # Force strength for overlap
    boundary_coeff = 20.0 # Force strength for boundaries
    damping = 0.9 # Velocity damping
    
    velocities = np.zeros_like(centers)
    
    # Annealing pressure to find the limit
    # Start with lower pressure, increase it?
    # Actually, we want to find the state where pressure > 0 forces expansion,
    # but overlap forces contraction.
    # A better way: Fix sum of radii target? No, we maximize it.
    # Let's use a penalty method implicitly.
    
    # We will run a loop where we try to expand radii.
    # If overlap occurs, we resolve it by moving centers.
    # If no overlap, we increase radii.
    
    # This is similar to finding the "equilibrium" of a gas in a box with variable particle sizes.
    
    # Let's refine the loop structure:
    # 1. Check for overlaps and boundary violations.
    # 2. Apply forces to resolve violations (move centers).
    # 3. If system is valid (low violation), try to increase radii slightly.
    # 4. If system is invalid, do not increase radii (or decrease them).
    
    # However, continuous increase is hard. 
    # Let's use a simplified energy minimization with a target radius sum?
    # No, let's just use the repulsion dynamics with a "growth" force.
    
    # Force model:
    # F_ij = repulsion if dist < r_i + r_j
    # F_boundary = repulsion if out of bounds
    # F_growth = outward force on radii?
    # Actually, we can treat radii as dynamic variables.
    # dr/dt = alpha * (1 - overlap_penalty)
    
    # Let's implement a discrete step simulation.
    
    current_pressure = 0.0001
    temp = 1.0 # Temperature for random jitter
    
    for step in range(n_iter):
        # Decay temperature
        if step > 1000:
            temp *= 0.999
        
        # Update radii based on "validity"
        # Calculate total overlap
        total_overlap = 0.0
        forces = np.zeros_like(centers)
        r_forces = np.zeros(N)
        
        # 1. Calculate pair interactions
        for i in range(N):
            for j in range(i + 1, N):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                r_sum = radii[i] + radii[j]
                
                if dist < r_sum and dist > 1e-9:
                    overlap = r_sum - dist
                    total_overlap += overlap**2
                    
                    # Normalize vector
                    n_vec = dist_vec / dist
                    
                    # Repulsive force: push apart
                    # Force magnitude proportional to overlap
                    f_mag = overlap * repulsion_coeff
                    
                    # Apply to centers
                    forces[i] -= f_mag * n_vec
                    forces[j] += f_mag * n_vec
                    
                    # Pressure on radii: if overlap, shrink radii?
                    # Or just let position adjustment handle it.
                    # Let's try to shrink radii proportional to overlap contribution
                    r_forces[i] -= overlap
                    r_forces[j] -= overlap
                    
        # 2. Calculate boundary interactions
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            
            # Left boundary
            if x < r:
                overlap = r - x
                total_overlap += overlap**2
                forces[i, 0] += boundary_coeff * overlap
                r_forces[i] -= boundary_coeff * overlap * 0.5 # Shrink if hitting wall
                
            # Right boundary
            if x > 1.0 - r:
                overlap = (1.0 - r) - x # x is too big
                # Actually violation amount is x - (1-r)
                overlap = x - (1.0 - r)
                total_overlap += overlap**2
                forces[i, 0] -= boundary_coeff * overlap
                r_forces[i] -= boundary_coeff * overlap * 0.5
                
            # Bottom boundary
            if y < r:
                overlap = r - y
                total_overlap += overlap**2
                forces[i, 1] += boundary_coeff * overlap
                r_forces[i] -= boundary_coeff * overlap * 0.5
                
            # Top boundary
            if y > 1.0 - r:
                overlap = y - (1.0 - r)
                total_overlap += overlap**2
                forces[i, 1] -= boundary_coeff * overlap
                r_forces[i] -= boundary_coeff * overlap * 0.5
                
        # 3. Update positions (Velocity Verlet style or simple Euler)
        # Add random jitter for exploration
        noise = np.random.normal(0, temp * 0.001, size=centers.shape)
        
        # Update velocities
        velocities = damping * velocities + forces + noise
        
        # Update centers
        centers += velocities
        
        # 4. Update radii
        # If overlap is low, expand radii. If high, shrink.
        # We want to maximize sum(r).
        # Gradient of sum(r) is 1.
        # Penalty gradient is -r_forces.
        
        # Simple rule:
        if total_overlap < 1e-6:
            # System is valid (mostly), expand radii
            expansion_rate = 0.0005 * (1 + 0.1 * math.sin(step * 0.1)) # Oscillate to avoid sticking
            radii += expansion_rate
        else:
            # System has overlap, shrink radii to resolve
            # Shrink proportional to overlap severity
            shrink_rate = total_overlap * 0.01
            radii -= shrink_rate
            
        # Ensure radii are non-negative
        radii = np.maximum(radii, 1e-5)
        
        # Keep centers inside [0,1] strictly to avoid numerical issues
        # (Forces should handle this, but clamp just in case)
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
        # Safety check: if radii get too big (impossible), clamp?
        # Not really needed if logic is sound.

    # 3. Final Polish with Gradient Ascent on Sum of Radii
    # Use a simple local search to tighten constraints and maximize sum.
    # Since we might be close, we can try to push radii up and fix overlaps.
    
    # Check if we are valid
    # If not, reduce radii until valid
    valid = False
    while not valid:
        # Check overlaps
        max_ov = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j]:
                    max_ov = max(max_ov, radii[i] + radii[j] - dist)
            # Check boundaries
            if centers[i, 0] < radii[i]: max_ov = max(max_ov, radii[i] - centers[i, 0])
            if centers[i, 0] > 1 - radii[i]: max_ov = max(max_ov, centers[i, 0] - (1 - radii[i]))
            if centers[i, 1] < radii[i]: max_ov = max(max_ov, radii[i] - centers[i, 1])
            if centers[i, 1] > 1 - radii[i]: max_ov = max(max_ov, centers[i, 1] - (1 - radii[i]))
            
        if max_ov > 1e-7:
            # Reduce radii slightly to fix overlap
            radii -= max_ov * 0.5
        else:
            valid = True
            
    # Now try to expand radii slightly and re-optimize positions locally
    for _ in range(50):
        # Try to increase radii
        radii *= 1.001
        
        # Resolve overlaps by moving centers (repulsion)
        for _ in range(100): # Inner iterations to resolve
            moved = False
            for i in range(N):
                # Check overlap with others
                for j in range(i + 1, N):
                    dist_vec = centers[i] - centers[j]
                    dist = np.linalg.norm(dist_vec)
                    r_sum = radii[i] + radii[j]
                    if dist < r_sum and dist > 1e-9:
                        overlap = r_sum - dist
                        n_vec = dist_vec / dist
                        # Move i away from j
                        move = overlap * 0.5 * n_vec
                        centers[i] -= move
                        centers[j] += move
                        moved = True
                
                # Check boundaries
                # Left
                if centers[i, 0] < radii[i]:
                    centers[i, 0] = radii[i]
                    moved = True
                # Right
                if centers[i, 0] > 1.0 - radii[i]:
                    centers[i, 0] = 1.0 - radii[i]
                    moved = True
                # Bottom
                if centers[i, 1] < radii[i]:
                    centers[i, 1] = radii[i]
                    moved = True
                # Top
                if centers[i, 1] > 1.0 - radii[i]:
                    centers[i, 1] = 1.0 - radii[i]
                    moved = True
            
            if not moved:
                break
        
        # Verify validity
        is_valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    is_valid = False
            if centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1 - radii[i] + 1e-9: is_valid = False
            if centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1 - radii[i] + 1e-9: is_valid = False
            
        if not is_valid:
            # Failed to resolve, shrink radii back
            radii *= 0.995
        # Else: kept new radii

    return centers, radii, np.sum(radii)

# Helper to check validity for debugging (not part of solution return)
def check(centers, radii):
    import numpy as np
    n = centers.shape[0]
    # Check for NaN values
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0: return False
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

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {check(centers, radii)}")
    print(f"Min Radius: {np.min(radii)}, Max Radius: {np.max(radii)}")

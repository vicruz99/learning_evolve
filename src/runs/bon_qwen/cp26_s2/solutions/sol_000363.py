# sol_000363 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b4024b4) state=3482d5d4 sum of radii=2.593169 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing():
    """
    Attempt to pack 26 circles in a unit square to maximize sum of radii.
    Uses a combination of initialization and optimization.
    """
    n = 26
    
    # Helper to calculate distance
    def dist(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    # Helper to check validity
    def check_valid(centers, radii):
        # Boundary
        for i in range(n):
            if radii[i] < 0: return False
            if centers[i][0] - radii[i] < -1e-7 or centers[i][0] + radii[i] > 1 + 1e-7: return False
            if centers[i][1] - radii[i] < -1e-7 or centers[i][1] + radii[i] > 1 + 1e-7: return False
        # Overlaps
        for i in range(n):
            for j in range(i + 1, n):
                d = dist(centers[i], centers[j])
                if d < radii[i] + radii[j] - 1e-7:
                    return False
        return True

    # Objective function to maximize: sum of radii
    # We will minimize the negative sum of radii
    def objective(params):
        # params: [x0, y0, r0, x1, y1, r1, ...]
        # But we need to ensure constraints are handled.
        # A penalty approach might be easier for simple optimization.
        return -np.sum(params[2::3])

    # Constraints are hard. Let's use a force-directed simulation first to find a good config,
    # then refine with scipy.

    # Initialization: 5x5 grid + 1
    # 5x5 grid centers at 0.1, 0.3, 0.5, 0.7, 0.9
    # But we have 26 circles.
    # Let's try a dense random packing and then optimize.
    
    np.random.seed(42)
    
    # Initialize centers in a grid-like pattern but perturbed
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05 # Start small
    
    # Grid placement
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                centers[idx] = [0.1 + i*0.2 + np.random.uniform(-0.01, 0.01), 
                                0.1 + j*0.2 + np.random.uniform(-0.01, 0.01)]
                idx += 1
    if idx < n:
        centers[idx] = [0.5, 0.5] # Center for the last one
    
    # Simulation parameters
    force_strength = 0.1
    repulsion_strength = 10.0
    boundary_strength = 5.0
    expansion_rate = 0.001
    
    # Simulate
    for step in range(2000):
        # Increase radii slightly
        for i in range(n):
            radii[i] += expansion_rate * (1.0 - radii[i]) # Expand towards 1, but constrained by others
        
        forces = np.zeros((n, 2))
        
        # Calculate repulsive forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                vec = centers[i] - centers[j]
                d = np.linalg.norm(vec)
                if d == 0:
                    d = 1e-9
                    vec = [1.0, 0.0]
                
                r_sum = radii[i] + radii[j]
                
                # If overlapping, strong repulsion
                if d < r_sum:
                    overlap = r_sum - d
                    # Force proportional to overlap
                    force_mag = repulsion_strength * overlap / d
                    f = force_mag * vec / d # Direction i->j is vec, but we want to push apart
                    # Actually vec is i - j. Pushing i away from j means force in direction vec.
                    # Wait, if d < r_sum, they overlap. We want to move i away from j.
                    # Vector from j to i is vec. So force on i is along vec.
                    forces[i] += f
                    forces[j] -= f
                else:
                    # Soft repulsion to keep them apart slightly? 
                    # Not strictly necessary for validity but helps packing
                    pass
        
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i][0] += boundary_strength * (r - x) # Push right?
                # If x - r < 0, we need to increase x. Force should be positive.
                # r - x is positive. So yes.
            # Right wall
            if x + r > 1:
                forces[i][0] -= boundary_strength * (x + r - 1) # Push left
            # Bottom wall
            if y - r < 0:
                forces[i][1] += boundary_strength * (r - y)
            # Top wall
            if y + r > 1:
                forces[i][1] -= boundary_strength * (y + r - 1)
                
            # Also, if circle is too close to boundary without overlapping, 
            # it limits radius. But here we just push it in if it's out.
            # To maximize radius, we want to be as far from boundary as possible?
            # No, we want to be constrained by boundaries to allow large radius?
            # Actually, if a circle is in the middle, its max radius is 0.5.
            # If it's near corner, max radius is smaller.
            # But we have many circles.
            
        # Update centers
        # Damping to prevent oscillation
        damping = 0.5
        centers += forces * force_strength * (1.0 - step/2000.0) * damping
        
        # Clamp centers to [0, 1]
        centers = np.clip(centers, 0, 1)
        
        # Re-check radii validity and reduce if overlapping
        # This is a "shrink if necessary" step
        valid_r = True
        while valid_r:
            valid_r = False
            for i in range(n):
                # Check boundaries
                r_max = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
                if radii[i] > r_max + 1e-9:
                    radii[i] = r_max
                    valid_r = True # Changed
                
                # Check neighbors
                for j in range(i + 1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    r_max_pair = d - radii[j]
                    if radii[i] > r_max_pair + 1e-9:
                        radii[i] = r_max_pair
                        valid_r = True
                        break # Restart loop for safety? Or just continue
        
        # Ensure non-negative radii
        radii = np.maximum(radii, 0.0)

    # After simulation, we might have a valid packing.
    # Try to optimize with scipy to polish.
    
    # Flatten params
    params = np.zeros(n * 3)
    for i in range(n):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = radii[i]
        
    # Objective: maximize sum of radii -> minimize negative sum
    def obj(params):
        r_sum = 0
        for i in range(n):
            r_sum += params[3*i+2]
        return -r_sum

    # Constraints
    cons = []
    
    # Boundary constraints
    for i in range(n):
        # x - r >= 0 => x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
        # r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_con(p, i=i, j=j):
                x1, y1, r1 = p[3*i], p[3*i+1], p[3*i+2]
                x2, y2, r2 = p[3*j], p[3*j+1], p[3*j+2]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                # dist >= r1 + r2  =>  dist^2 >= (r1+r2)^2
                # But dist^2 - (r1+r2)^2 >= 0 is not convex and can be tricky.
                # Better to use dist >= r1 + r2 directly?
                # But sqrt is slow.
                # Let's use the squared form carefully.
                # If dist^2 < (r1+r2)^2, constraint violated.
                # We need dist^2 - (r1+r2)^2 >= 0.
                # However, if r1+r2 is large, this is sensitive.
                # Let's stick to distance.
                dist = math.sqrt(dist_sq)
                return dist - (r1 + r2)
            
            cons.append({'type': 'ineq', 'fun': overlap_con})

    # Bounds
    bnds = []
    for i in range(n):
        bnds.append((0, 1)) # x
        bnds.append((0, 1)) # y
        bnds.append((0, 1)) # r (loose upper bound)

    # Use SLSQP
    try:
        res = minimize(obj, params, method='SLSQP', bounds=bnds, constraints=cons, 
                       options={'maxiter': 100, 'ftol': 1e-9})
        if res.success:
            params = res.x
    except Exception as e:
        pass # Fallback to simulation result

    # Extract final results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i][0] = params[3*i]
        final_centers[i][1] = params[3*i+1]
        final_radii[i] = params[3*i+2]
        
    # Final cleanup: clamp and ensure validity
    # If scipy failed or produced invalid, fallback to simulation
    if not check_valid(final_centers, final_radii):
        # Re-run simulation logic to fix validity if needed
        # Simple projection
        for i in range(n):
            # Clamp radii to boundary limits
            r_max = min(final_centers[i][0], 1 - final_centers[i][0], 
                        final_centers[i][1], 1 - final_centers[i][1])
            final_radii[i] = min(final_radii[i], r_max)
        
        # Check overlaps and shrink
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(final_centers[i] - final_centers[j])
                    if d < final_radii[i] + final_radii[j] - 1e-9:
                        # Shrink larger circle or both
                        r_sum = final_radii[i] + final_radii[j]
                        ratio = d / r_sum
                        final_radii[i] *= ratio
                        final_radii[j] *= ratio # Shrink both equally? Or just one.
                        # Shrinking both reduces sum more.
                        # Better to shrink the one that is "less constrained"?
                        # Heuristic: shrink the one with larger radius?
                        if final_radii[i] >= final_radii[j]:
                            final_radii[i] = d - final_radii[j]
                        else:
                            final_radii[j] = d - final_radii[i]
                        changed = True
                        break # Restart
                if changed: break
            
            # Re-check boundary after shrinking? Radii only decreased, so boundary still ok.
            
    # Final clamp radii to 0
    final_radii = np.maximum(final_radii, 0.0)
    
    return final_centers, final_radii, np.sum(final_radii)

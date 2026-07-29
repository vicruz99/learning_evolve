# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state aba87625) state=a36afecc sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (centers, radii, sum_radii) for a packing of 26 circles in a unit square.
    """
    np.random.seed(42)
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Initial hexagonal-like placement
    # 5 rows: 5, 4, 5, 4, 5 = 23 circles. Add 3 more? 
    # Let's try a denser 5x5 grid + 1 offset or similar
    # We'll use a 6-row staggered pattern: 4, 5, 4, 5, 4, 4 = 26
    row_counts = [4, 5, 4, 5, 4, 4]
    idx = 0
    for r_idx, count in enumerate(row_counts):
        y = 0.1 + r_idx * 0.16  # Initial vertical spacing
        row_width = (count - 1) * 0.2
        start_x = (1.0 - row_width) / 2
        if r_idx % 2 == 1:
            start_x += 0.1  # Shift for staggering
        for c in range(count):
            if idx < n:
                centers[idx] = [start_x + c * 0.2, y]
                radii[idx] = 0.09
                idx += 1

    # Optimization parameters
    dt = 0.01
    friction = 0.8
    expansion_rate = 0.0005
    max_iter = 5000
    cooling_rate = 0.9995

    # Optimization loop
    for _ in range(max_iter):
        forces = np.zeros_like(centers)
        
        # 1. Compute repulsive forces for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap force
                    overlap = min_dist - dist
                    force_magnitude = overlap / dt
                    direction = diff / dist
                    forces[i] += direction * force_magnitude * 0.5
                    forces[j] -= direction * force_magnitude * 0.5
        
        # 2. Boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left boundary
            if x < r:
                forces[i, 0] += (r - x) / dt
            # Right boundary
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) / dt
            # Bottom boundary
            if y < r:
                forces[i, 1] += (r - y) / dt
            # Top boundary
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) / dt

        # Apply forces to move centers
        centers += forces * dt
        centers *= friction  # Simple friction/damping
        
        # Keep centers strictly inside [0, 1] to prevent drift issues
        centers = np.clip(centers, 1e-5, 1 - 1e-5)

        # Expand radii slightly if no major overlaps (heuristic check)
        # We check a few random pairs to estimate stability
        # If stable, expand. If not, expand slower.
        overlaps = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j]:
                    overlaps += 1
                    break
            if overlaps > 0:
                break
        
        if overlaps == 0:
            # No overlaps detected in sample, try to expand
            # Expand more if we are far from boundary constraints
            expand = expansion_rate * cooling_rate**(_ // 100)
            # Limit expansion by distance to boundary
            for i in range(n):
                r = radii[i]
                x, y = centers[i]
                max_r = min(x, 1-x, y, 1-y)
                if max_r > r:
                    radii[i] += expand
                else:
                    radii[i] = max_r
        else:
            # Overlaps exist, shrink slightly or hold
            pass

    # Final refinement: ensure strict validity
    # If any overlap remains, reduce radii slightly
    valid = False
    while not valid:
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                req_dist = radii[i] + radii[j]
                if dist < req_dist - 1e-12:
                    # Reduce radii equally to resolve
                    diff = req_dist - dist
                    delta = diff / 2
                    radii[i] -= delta
                    radii[j] -= delta
                    valid = False
        
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if r > x - 1e-12:
                radii[i] = x - 1e-12
                valid = False
            if r > (1 - x) - 1e-12:
                radii[i] = (1 - x) - 1e-12
                valid = False
            if r > y - 1e-12:
                radii[i] = y - 1e-12
                valid = False
            if r > (1 - y) - 1e-12:
                radii[i] = (1 - y) - 1e-12
                valid = False
            
            # Ensure non-negative
            if radii[i] < 0:
                radii[i] = 0

    return centers, radii, np.sum(radii)

if __name__ == "__main__":
    import sys
    # Allow the script to run and print results if executed directly
    # This part is for testing, not required for the function definition
    try:
        import numpy as np
        centers, radii, sum_r = run_packing()
        print(f"Sum of radii: {sum_r:.5f}")
        
        # Validate
        def validate_packing(centers, radii):
            n = centers.shape[0]
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

        if validate_packing(centers, radii):
            print("Validation: PASSED")
        else:
            print("Validation: FAILED")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

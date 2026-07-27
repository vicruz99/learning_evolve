import numpy as np

def run_packing():
    """
    Returns (centers, radii, sum_radii) for a packing of 26 circles 
    in a unit square that maximizes the sum of radii.
    """
    np.random.seed(42)
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 1. Initialization: 5x5 Grid + 1 Center
    grid_indices = []
    for r in range(5):
        for c in range(5):
            idx = r * 5 + c
            centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
            radii[idx] = 0.1
            grid_indices.append(idx)

    # Place 26th circle in the center gap
    centers[25] = [0.5, 0.5]
    radii[25] = 0.02

    # 2. Force-directed expansion
    expansion_rate = 1.0005
    max_iterations = 5000
    
    for _ in range(max_iterations):
        # Expand radii
        radii *= expansion_rate
        
        # Check boundary and shrink
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Clamping to boundaries
            if x - r < 0:
                centers[i, 0] = r
            elif x + r > 1:
                centers[i, 0] = 1 - r
            if y - r < 0:
                centers[i, 1] = r
            elif y + r > 1:
                centers[i, 1] = 1 - r

        # Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Calculate push distance
                    overlap = min_dist - dist
                    total_r = radii[i] + radii[j]
                    push_i = overlap * (radii[j] / total_r) * 0.5
                    push_j = overlap * (radii[i] / total_r) * 0.5
                    
                    # Normalize direction
                    ndx = dx / dist
                    ndy = dy / dist
                    
                    # Apply push
                    centers[i, 0] -= ndx * push_i
                    centers[i, 1] -= ndy * push_i
                    centers[j, 0] += ndx * push_j
                    centers[j, 1] += ndy * push_j

        # Re-verify boundaries after movement
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: centers[i, 0] = r
            if x + r > 1: centers[i, 0] = 1 - r
            if y - r < 0: centers[i, 1] = r
            if y + r > 1: centers[i, 1] = 1 - r

    # 3. Final Refinement
    for _ in range(200):
        for i in range(n):
            can_expand = True
            # Check against boundaries
            max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            if radii[i] >= max_r - 1e-9:
                can_expand = False
            
            if can_expand:
                # Check against other circles
                for j in range(n):
                    if i != j:
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        allowed_r = dist - radii[j]
                        if radii[i] >= allowed_r - 1e-9:
                            can_expand = False
                            break
            
            if can_expand:
                # Expand until hitting first constraint
                r_limit = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
                for j in range(n):
                    if i != j:
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        r_limit = min(r_limit, dist - radii[j])
                
                if r_limit > radii[i]:
                    radii[i] = r_limit

    # 4. Final Validation and cleanup
    radii = np.maximum(radii, 0)
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# To validate:
# import numpy as np
# def validate_packing(centers, radii):
#     ... (from prompt)
# ...
# c, r, s = run_packing()
# print("Sum of radii:", s)
# print("Valid:", validate_packing(c, r))
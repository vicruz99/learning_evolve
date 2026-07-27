import numpy as np

def compute_forces(centers, r, n):
    force = np.zeros_like(centers)
    # Boundary forces
    for i in range(n):
        c = centers[i]
        if c[0] < r: 
            force[i, 0] += (r - c[0]) * 20.0
        if c[0] > 1.0 - r: 
            force[i, 0] -= (c[0] - (1.0 - r)) * 20.0
        if c[1] < r: 
            force[i, 1] += (r - c[1]) * 20.0
        if c[1] > 1.0 - r: 
            force[i, 1] -= (c[1] - (1.0 - r)) * 20.0

    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist_sq = np.sum(diff**2)
            dist = np.sqrt(dist_sq)
            if dist < 2.0 * r:
                if dist < 1e-12:
                    force[i] += np.random.rand(2) - 0.5
                    force[j] -= np.random.rand(2) - 0.5
                else:
                    direction = diff / max(dist, 1e-9)
                    overlap = 2.0 * r - dist
                    push = overlap * 15.0
                    force[i] += direction * push
                    force[j] -= direction * push
    return force

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Hexagonal initialization
    centers = np.zeros((n, 2))
    idx = 0
    r_start = 0.08
    dx = 2.0 * r_start
    dy = np.sqrt(3) * r_start
    
    y = r_start
    row = 0
    while idx < n:
        x = r_start + (row % 2) * dx / 2.0
        while x <= 1.0 - r_start and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += dx
        y += dy
        row += 1
        
    # Add slight random perturbation to break symmetry
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    
    # Clip to valid range initially
    centers[:, 0] = np.clip(centers[:, 0], r_start, 1.0 - r_start)
    centers[:, 1] = np.clip(centers[:, 1], r_start, 1.0 - r_start)
        
    r = r_start
    lr = 0.05
    
    # Expansion and relaxation phase
    for step in range(4000):
        r += 0.00012
        lr *= 0.9985
        
        forces = compute_forces(centers, r, n)
        centers += forces * lr
        
        # Hard boundary constraints
        centers[:, 0] = np.clip(centers[:, 0], r, 1.0 - r)
        centers[:, 1] = np.clip(centers[:, 1], r, 1.0 - r)
        
    # Final validity adjustment
    for _ in range(500):
        valid = True
        
        # Check boundaries
        for i in range(n):
            c = centers[i]
            if c[0] < r or c[0] > 1.0 - r or c[1] < r or c[1] > 1.0 - r:
                valid = False
                break
        
        if not valid:
            r -= 0.0002
            centers[:, 0] = np.clip(centers[:, 0], r, 1.0 - r)
            centers[:, 1] = np.clip(centers[:, 1], r, 1.0 - r)
            continue
            
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < 2.0 * r - 1e-11:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            break
        r -= 0.0002
        
    radii = np.full(n, r)
    return centers, radii, np.sum(radii)
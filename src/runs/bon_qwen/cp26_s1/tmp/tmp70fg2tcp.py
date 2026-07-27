import numpy as np

def compute_forces(centers, radii, N):
    """Compute repulsive forces for overlaps and boundary violations."""
    forces = np.zeros_like(centers)
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                if dist < 1e-8:
                    dist = 1e-8
                    diff = np.array([0.001, 0.0])
                overlap = min_dist - dist
                force_dir = diff / dist
                forces[i] += overlap * force_dir
                forces[j] -= overlap * force_dir
                
    for i in range(N):
        for d in range(2):
            if centers[i, d] < radii[i]:
                forces[i, d] += (radii[i] - centers[i, d]) * 5.0
            elif centers[i, d] > 1.0 - radii[i]:
                forces[i, d] -= (centers[i, d] - (1.0 - radii[i])) * 5.0
    return forces

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    N = 26
    
    # Hexagonal-ish initialization for better convergence
    centers = np.zeros((N, 2))
    idx = 0
    y = 0.2
    while idx < N and y < 1.0:
        x = 0.2
        shift = (int(y / 0.2) % 2) * 0.1
        while x < 1.0 and idx < N:
            centers[idx] = [x + shift, y]
            x += 0.2
            idx += 1
        y += 0.173205 # sqrt(3)/2 * 0.2
        if idx >= N: break
        
    # Add small perturbation to break symmetry
    centers += np.random.randn(N, 2) * 0.001
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.ones(N) * 0.005
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = 0.0
    
    for step in range(6000):
        # Slowly inflate radii to drive packing density
        radii *= 1.0002
        
        # Compute forces based on current overlaps and boundary constraints
        forces = compute_forces(centers, radii, N)
        
        # Adaptive learning rate
        lr = 0.05 / (1.0 + step * 0.001)
        centers += lr * forces
        
        # Keep centers within bounds
        centers = np.clip(centers, 0.0, 1.0)
        
        # Validate configuration
        valid = True
        for i in range(N):
            if centers[i,0] < radii[i] - 1e-9 or centers[i,0] > 1 - radii[i] + 1e-9:
                valid = False; break
            if centers[i,1] < radii[i] - 1e-9 or centers[i,1] > 1 - radii[i] + 1e-9:
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False; break
                if not valid: break
                
        if valid:
            s = np.sum(radii)
            if s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    return best_centers, best_radii, float(best_sum)
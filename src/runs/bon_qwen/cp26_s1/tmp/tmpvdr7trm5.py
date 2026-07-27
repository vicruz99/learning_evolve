import numpy as np

def compute_valid_radii(centers):
    """
    Given a set of centers, compute the maximum valid radius for each circle
    such that they fit in [0,1]x[0,1] and do not overlap.
    """
    n = centers.shape[0]
    radii = np.full(n, 1.0)
    
    # Wall constraints
    left = centers[:, 0]
    right = 1.0 - centers[:, 0]
    bottom = centers[:, 1]
    top = 1.0 - centers[:, 1]
    radii = np.minimum(radii, left)
    radii = np.minimum(radii, right)
    radii = np.minimum(radii, bottom)
    radii = np.minimum(radii, top)
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            max_r = dist / 2.0
            if max_r < radii[i]:
                radii[i] = max_r
            if max_r < radii[j]:
                radii[j] = max_r
                
    # Add small epsilon to strictly satisfy validator tolerance
    radii -= 1e-8
    return np.maximum(radii, 0.0)

def simulate_packing(centers_init, n, steps=3000, growth_rate=1.0015, dt=0.015, friction=0.92):
    """
    Run a force-directed simulation to spread circles and allow radii expansion.
    """
    centers = centers_init.copy()
    radii = np.full(n, 0.005)
    vel = np.zeros_like(centers)
    
    # Pre-allocate indices for pairwise interactions to speed up loop
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(steps):
        radii *= growth_rate
        
        forces = np.zeros_like(centers)
        
        # Wall forces (strong repulsion)
        for i in range(n):
            cx, cy = centers[i]
            r = radii[i]
            if cx - r < 0: forces[i, 0] += (r - cx) * 2000.0
            if cx + r > 1: forces[i, 0] -= (cx + r - 1) * 2000.0
            if cy - r < 0: forces[i, 1] += (r - cy) * 2000.0
            if cy + r > 1: forces[i, 1] -= (cy + r - 1) * 2000.0
            
        # Pairwise forces
        for idx in range(len(i_idx)):
            i, j = i_idx[idx], j_idx[idx]
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                # Force proportional to overlap
                fx = (overlap / dist) * dx * 1500.0
                fy = (overlap / dist) * dy * 1500.0
                forces[i, 0] -= fx
                forces[i, 1] -= fy
                forces[j, 0] += fx
                forces[j, 1] += fy
                
        # Integration step
        vel += forces * dt
        vel *= friction
        centers += vel * dt
        
        # Hard clip to boundaries to prevent explosion
        centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
        
    return centers

def local_search_refine(centers, n, iterations=500, step_size=0.005):
    """
    Iteratively perturb centers to find configurations that allow larger radii.
    """
    current_radii = compute_valid_radii(centers)
    current_sum = np.sum(current_radii)
    best_centers = centers.copy()
    best_radii = current_radii.copy()
    
    rng = np.random.RandomState(123)
    
    for _ in range(iterations):
        # Randomly pick a circle to move
        i = rng.randint(n)
        # Try random direction
        move = rng.uniform(-step_size, step_size, 2)
        
        test_centers = centers.copy()
        test_centers[i] += move
        test_centers = np.clip(test_centers, 1e-5, 1.0 - 1e-5)
        
        # Only compute radii for the affected circle and its neighbors for speed
        # But for correctness with n=26, full compute is cheap enough
        test_radii = compute_valid_radii(test_centers)
        test_sum = np.sum(test_radii)
        
        if test_sum > current_sum:
            centers = test_centers
            current_radii = test_radii
            current_sum = test_sum
            
            if current_sum > np.sum(best_radii):
                best_centers = centers.copy()
                best_radii = current_radii.copy()
                
        # Adaptive step size
        step_size *= 0.995
        
    return best_centers, best_radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Run multiple simulations with different seeds/initializations
    for seed in range(15):
        rng = np.random.RandomState(seed)
        
        # Initialize with hexagonal-ish lattice perturbed by noise
        centers = np.zeros((n, 2))
        count = 0
        y = 0.05
        while count < n:
            x = 0.05
            row_idx = count // 5
            shift = (row_idx % 2) * 0.05
            while x < 0.95 and count < n:
                centers[count, 0] = x + shift + rng.normal(0, 0.005)
                centers[count, 1] = y + rng.normal(0, 0.005)
                count += 1
                x += 0.15
            y += 0.12
        centers = centers[:n]
        
        # Run simulation
        sim_centers = simulate_packing(centers, n, steps=4000)
        
        # Project to valid radii
        sim_radii = compute_valid_radii(sim_centers)
        sim_sum = np.sum(sim_radii)
        
        # Local search refinement
        refined_centers, refined_radii = local_search_refine(sim_centers, n, iterations=800)
        refined_sum = np.sum(refined_radii)
        
        if refined_sum > best_sum:
            best_sum = refined_sum
            best_centers = refined_centers.copy()
            best_radii = refined_radii.copy()
            
    # Final safety cleanup
    best_radii = compute_valid_radii(best_centers)
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
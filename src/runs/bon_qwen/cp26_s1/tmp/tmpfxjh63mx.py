import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Uses a growing circles algorithm with multiple restarts.
    """
    n = 26
    best_sum = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)

    def solve_from_seed(initial_centers):
        centers = initial_centers.copy()
        radii = np.full(n, 0.05)  # Start with small valid radius
        
        # Phase 1: Grow radii and resolve collisions
        for step in range(2000):
            # Increase radii
            dr = 0.00005
            radii += dr
            
            # Resolve collisions and boundaries
            for _ in range(100): # Iterative relaxation
                moved = False
                # Boundary constraints
                for i in range(n):
                    r = radii[i]
                    if centers[i, 0] - r < 0:
                        centers[i, 0] = r
                        moved = True
                    if centers[i, 0] + r > 1:
                        centers[i, 0] = 1 - r
                        moved = True
                    if centers[i, 1] - r < 0:
                        centers[i, 1] = r
                        moved = True
                    if centers[i, 1] + r > 1:
                        centers[i, 1] = 1 - r
                        moved = True
                
                # Pairwise overlaps
                for i in range(n):
                    for j in range(i + 1, n):
                        diff = centers[j] - centers[i]
                        dist = np.sqrt(np.sum(diff ** 2))
                        min_dist = radii[i] + radii[j]
                        
                        if dist < min_dist and dist > 1e-12:
                            # Normalize direction
                            dir_vec = diff / dist
                            overlap = min_dist - dist
                            # Move apart by half the overlap
                            shift = 0.5 * overlap * dir_vec
                            centers[i] -= shift
                            centers[j] += shift
                            moved = True
                        elif dist < 1e-12:
                            # If centers are identical, push randomly
                            centers[i] -= 0.001
                            centers[j] += 0.001
                            moved = True
                            
                if not moved:
                    break
        
        return centers, radii, np.sum(radii)

    # Seed 1: Grid arrangement (5x5 + 1 center)
    c1 = []
    for r in range(5):
        for c in range(5):
            c1.append([0.1 + c * 0.2, 0.1 + r * 0.2])
    c1.append([0.5, 0.5]) # 26th circle
    c1 = np.array(c1)
    
    # Seed 2: Hexagonal lattice
    c2 = []
    r = 0.06
    for row in range(7):
        y = r + row * r * np.sqrt(3)
        cols = 4 if row % 2 == 0 else 3
        for col in range(cols):
            x = r + col * 2 * r
            c2.append([x, y])
            if len(c2) >= 26:
                break
        if len(c2) >= 26:
            break
    c2 = np.array(c2[:26])

    # Seed 3: Random
    np.random.seed(42)
    c3 = np.random.rand(n, 2) * 0.8 + 0.1

    # Seed 4: Random perturbation of grid
    c4 = c1 + np.random.randn(*c1.shape) * 0.02
    
    seeds = [c1, c2, c3, c4]
    
    for seed in seeds:
        try:
            centers, radii, s = solve_from_seed(seed)
            if s > best_sum:
                best_sum = s
                best_centers = centers
                best_radii = radii
        except Exception:
            continue

    # Final validation clamp
    for i in range(n):
        r = best_radii[i]
        best_centers[i, 0] = np.clip(best_centers[i, 0], r, 1 - r)
        best_centers[i, 1] = np.clip(best_centers[i, 1], r, 1 - r)

    return best_centers, best_radii, best_sum
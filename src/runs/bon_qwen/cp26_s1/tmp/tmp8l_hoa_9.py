import numpy as np

def run_packing():
    n = 26
    
    def force_optimize(init_centers, init_radii, seed=0):
        np.random.seed(seed)
        centers = init_centers.copy()
        radii = init_radii.copy()
        
        dt = 0.1
        grow = 0.001
        max_iter = 20000
        
        for step in range(max_iter):
            F = np.zeros((n, 2))
            
            # Pairwise repulsion forces
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            r_sum = radii[:, None] + radii[None, :]
            overlap = r_sum - dist
            overlap[overlap < 0] = 0.0
            
            norm_dir = diff / (dist[:, :, None] + 1e-12)
            F += np.sum(overlap[:, :, None] * norm_dir, axis=1)
            
            # Boundary repulsion forces
            F[:, 0] += np.maximum(0, radii - centers[:, 0])
            F[:, 0] -= np.maximum(0, centers[:, 0] + radii - 1)
            F[:, 1] += np.maximum(0, radii - centers[:, 1])
            F[:, 1] -= np.maximum(0, centers[:, 1] + radii - 1)
            
            centers += dt * F
            centers = np.clip(centers, 0.0, 1.0)
            
            # Compute total penalty
            pen = np.sum(overlap) + np.sum(np.maximum(0, radii - centers[:, 0])) + \
                  np.sum(np.maximum(0, centers[:, 0] + radii - 1)) + \
                  np.sum(np.maximum(0, radii - centers[:, 1])) + \
                  np.sum(np.maximum(0, centers[:, 1] + radii - 1))
            
            if pen < 1e-5:
                radii += grow
                # Occasional radius perturbation to explore non-uniform sizes
                if step % 2000 == 0 and step > 0:
                    radii += np.random.randn(n) * 0.002
            else:
                if pen > 0.05:
                    radii -= grow * 0.4
                # Position perturbation to escape local minima
                if step % 800 == 0:
                    centers += np.random.randn(n, 2) * 0.008
                    centers = np.clip(centers, 0.0, 1.0)
            
            # Annealing schedule
            if step % 50 == 0:
                dt *= 0.996
                grow *= 0.999
                
        return centers, radii

    # Triangular lattice initialization
    s = 0.11
    pts = []
    for j in range(12):
        for i in range(12):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    pts = pts * 0.85 + 0.075
    
    dists = np.sum((pts - 0.5)**2, axis=1)
    idx = np.argsort(dists)[:n]
    init_centers = pts[idx].copy()
    init_radii = np.full(n, 0.04)
    
    best_sum = -1.0
    best_out = (init_centers, init_radii)
    
    # Multiple restarts
    for seed in range(20):
        noise = np.random.rand(n, 2) * 0.01
        c, r = force_optimize(init_centers + noise, init_radii, seed=seed)
        r = np.maximum(r, 1e-9)
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_out = (c, r)
            
    centers, radii = best_out
    
    # Final constraint enforcement
    for _ in range(200):
        max_viol = 0.0
        max_viol = max(max_viol, np.max(np.maximum(0, radii - centers[:, 0])))
        max_viol = max(max_viol, np.max(np.maximum(0, centers[:, 0] + radii - 1)))
        max_viol = max(max_viol, np.max(np.maximum(0, radii - centers[:, 1])))
        max_viol = max(max_viol, np.max(np.maximum(0, centers[:, 1] + radii - 1)))
        
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        r_sum = radii[:, None] + radii[None, :]
        overlap = r_sum - dist
        triu_idx = np.triu_indices(n, k=1)
        max_viol = max(max_viol, np.max(overlap[triu_idx]))
        
        if max_viol < 1e-9:
            break
        radii -= max_viol + 1e-9
        
    radii = np.maximum(radii, 0.0)
    return centers, radii, np.sum(radii)
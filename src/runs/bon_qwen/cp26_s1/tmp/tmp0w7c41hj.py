import numpy as np
import math

def compute_gradients(centers, radii, alpha):
    """Compute gradients of the combined loss function w.r.t centers and radii."""
    n = centers.shape[0]
    d_c = np.zeros_like(centers)
    d_r = np.zeros_like(radii)
    
    # Inter-circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 1e-12:
                dist = 1e-12
            gap = radii[i] + radii[j] - dist
            if gap > 0:
                # Penalty term: gap^2
                # d(Penalty)/d_c_i = -2*gap * (c_i - c_j)/dist
                factor = -2.0 * gap / dist
                d_c[i] += factor * diff
                d_c[j] -= factor * diff
                # d(Penalty)/d_r_i = 2*gap
                d_r[i] += 2.0 * gap
                d_r[j] += 2.0 * gap

    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left boundary: r > x
        if r > x:
            diff = r - x
            d_r[i] += 2.0 * diff
            d_c[i, 0] -= 2.0 * diff
        # Right boundary: r > 1 - x
        if r > 1.0 - x:
            diff = r - (1.0 - x)
            d_r[i] += 2.0 * diff
            d_c[i, 0] += 2.0 * diff
        # Bottom boundary: r > y
        if r > y:
            diff = r - y
            d_r[i] += 2.0 * diff
            d_c[i, 1] -= 2.0 * diff
        # Top boundary: r > 1 - y
        if r > 1.0 - y:
            diff = r - (1.0 - y)
            d_r[i] += 2.0 * diff
            d_c[i, 1] += 2.0 * diff

    # Combine with objective gradient: Loss = Penalty - alpha * sum(radii)
    # Gradient w.r.t radii becomes: d_r - alpha
    return d_c, d_r - alpha

def run_optimization(centers, radii, alpha, lr, steps):
    """Perform gradient descent optimization."""
    for _ in range(steps):
        d_c, d_r = compute_gradients(centers, radii, alpha)
        
        # Gradient descent step
        centers -= lr * d_c
        radii -= lr * d_r
        
        # Hard constraints / clipping for stability
        centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
        radii = np.clip(radii, 1e-7, 0.49)
        
    return centers, radii

def get_hexagonal_init(n):
    """Generate a hexagonal lattice initialization."""
    c = np.zeros((n, 2))
    idx = 0
    r_est = 0.1
    y = r_est
    row_idx = 0
    
    while idx < n and y < 1.0 - r_est:
        x = r_est
        offset = r_est if row_idx % 2 == 1 else 0.0
        while x < 1.0 - r_est and idx < n:
            c[idx] = [x + offset, y]
            idx += 1
            x += 2.0 * r_est
        y += math.sqrt(3.0) * r_est
        row_idx += 1
        
    return c[:n]

def is_valid(centers, radii):
    """Check validity according to problem constraints."""
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1.0 + 1e-9 or y - r < -1e-9 or y + r > 1.0 + 1e-9:
            return False
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def project_to_valid(centers, radii):
    """Fallback: shrink radii uniformly until valid."""
    scale = 1.0
    for _ in range(100):
        if is_valid(centers, radii * scale):
            return centers, radii * scale
        scale *= 0.99
    return centers, radii * scale

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Initial configurations to try
    candidates = []
    
    # 1. Random initialization
    np.random.seed(42)
    c1 = np.random.rand(n, 2) * 0.8 + 0.1
    r1 = np.ones(n) * 0.02
    candidates.append((c1, r1))
    
    # 2. Grid initialization
    c2 = np.zeros((n, 2))
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < n:
                c2[idx] = [0.05 + j * 0.16, 0.05 + i * 0.16]
                idx += 1
    r2 = np.ones(n) * 0.02
    candidates.append((c2, r2))
    
    # 3. Hexagonal initialization
    c3 = get_hexagonal_init(n)
    r3 = np.ones(n) * 0.02
    candidates.append((c3, r3))
    
    # Optimization parameters
    alpha = 8.0
    lr = 1e-2
    
    for c_init, r_init in candidates:
        c_opt = c_init.copy()
        r_opt = r_init.copy()
        
        # Phase 1: Coarse optimization
        c_opt, r_opt = run_optimization(c_opt, r_opt, alpha, lr, 1500)
        
        # Phase 2: Fine optimization with lower learning rate
        c_opt, r_opt = run_optimization(c_opt, r_opt, alpha, lr * 0.1, 1000)
        
        # Ensure strict validity
        c_opt, r_opt = project_to_valid(c_opt, r_opt)
        
        current_sum = np.sum(r_opt)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
    return best_centers, best_radii, float(best_sum)
import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        for col in range(5):
            if idx < n:
                centers[idx] = [col * 0.2 + 0.1, row * 0.2 + 0.1]
                idx += 1
    
    lr = 0.05
    for _ in range(2000):
        # 1. Solve LP to find radii and marginals
        dists = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
        
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        pair_idx = []
        wall_info = []
        
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i], row[j] = 1.0, 1.0
                A_ub.append(row)
                b_ub.append(dists[i, j])
                pair_idx.append((i, j))
        
        for i in range(n):
            x, y = centers[i]
            for val in [x, 1-x, y, 1-y]:
                row = np.zeros(n)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(val)
                wall_info.append((i, val))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, method='highs')
        
        if res.success:
            radii = res.x
            marginals = res.ineqlin.marginals
            
            # 2. Compute gradients
            grad = np.zeros_like(centers)
            for k, (i, j) in enumerate(pair_idx):
                lam = marginals[k]
                if lam > 1e-6 and dists[i, j] > 1e-8:
                    force = lam * (centers[i] - centers[j]) / dists[i, j]
                    grad[i] += force
                    grad[j] -= force
            
            # Wall constraints
            for k in range(len(pair_idx), len(marginals)):
                lam = marginals[k]
                if lam > 1e-6:
                    i, val = wall_info[k]
                    if abs(val - centers[i, 0]) < 1e-6: grad[i, 0] += lam
                    elif abs(val - (1 - centers[i, 0])) < 1e-6: grad[i, 0] -= lam
                    elif abs(val - centers[i, 1]) < 1e-6: grad[i, 1] += lam
                    elif abs(val - (1 - centers[i, 1])) < 1e-6: grad[i, 1] -= lam
            
            # 3. Update centers
            centers += lr * grad
            centers = np.clip(centers, 1e-4, 1 - 1e-4)
            
        lr *= 0.999
        
    return centers, radii, float(np.sum(radii))

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
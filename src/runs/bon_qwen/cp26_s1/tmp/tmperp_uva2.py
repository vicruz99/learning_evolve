import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26


def hex_config(pattern, r):
    """Create hexagonal packing configuration."""
    c, r_list = [], []
    y = r
    for i, nc in enumerate(pattern):
        x0 = (1 - (nc - 1) * 2 * r) / 2
        if i % 2 == 1:
            x0 += r
        for j in range(nc):
            c.append([x0 + j * 2 * r, y])
            r_list.append(r)
        y += r * np.sqrt(3)
    return np.array(c), np.array(r_list)


def flatten_vars(v, n):
    """Flatten optimization variables into centers and radii."""
    return v[:2 * n].reshape(n, 2), v[2 * n:]


def objective(v, n):
    """Objective: minimize negative sum of radii."""
    _, rr = flatten_vars(v, n)
    return -np.sum(rr)


def constraints(v, n):
    """Boundary and non-overlap constraints."""
    cc, rr = flatten_vars(v, n)
    out = []
    # Boundary constraints: circle inside [0,1]x[0,1]
    for i in range(n):
        out.append(cc[i, 0] - rr[i])       # x - r >= 0
        out.append(1 - cc[i, 0] - rr[i])   # 1 - x - r >= 0
        out.append(cc[i, 1] - rr[i])       # y - r >= 0
        out.append(1 - cc[i, 1] - rr[i])   # 1 - y - r >= 0
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = cc[i, 0] - cc[j, 0]
            dy = cc[i, 1] - cc[j, 1]
            out.append(dx * dx + dy * dy - (rr[i] + rr[j]) ** 2)
    return np.array(out)


def optimize_pack(c, r, max_iter=5000):
    """Optimize packing from given initial configuration."""
    n = len(r)
    x0 = np.concatenate([c.flatten(), r])
    bnds = [(0, 1)] * 2 * n + [(1e-10, 0.5)] * n

    def obj(v):
        return objective(v, n)

    def cn(v):
        return constraints(v, n)

    cons = {'type': 'ineq', 'fun': cn}

    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-14})
        if res.success:
            co, ro = flatten_vars(res.x, n)
            return co, ro, np.sum(ro)
    except Exception:
        pass
    return c, r, np.sum(r)


def run_packing():
    """Run packing optimization and return best result."""
    n = N_CIRCLES
    best_sum = 0.0
    best_c = None
    best_r = None

    # Try different hexagonal patterns
    patterns = [
        [6, 5, 6, 5, 4],
        [7, 6, 7, 6],
        [5, 6, 5, 6, 4],
        [8, 6, 6, 6],
        [7, 7, 6, 6],
        [6, 7, 6, 7],
        [4, 7, 6, 7, 2],
        [9, 6, 6, 5],
        [5, 5, 5, 5, 6],
        [7, 5, 7, 7],
    ]

    for pat in patterns:
        if sum(pat) != n:
            continue
        # Try multiple starting radii
        for r_init in [0.06, 0.07, 0.08, 0.09]:
            c, r = hex_config(pat, r_init)
            co, ro, s = optimize_pack(c, r, max_iter=3000)
            if s > best_sum:
                best_sum = s
                best_c = co
                best_r = ro

    # Refine with random perturbations
    if best_c is not None:
        for i in range(15):
            np.random.seed(i * 137 + 42)
            cp = best_c + np.random.randn(n, 2) * 0.008
            rp = best_r + np.random.randn(n) * 0.003
            cp = np.clip(cp, 0.01, 0.99)
            rp = np.clip(rp, 0.001, 0.4)
            co, ro, s = optimize_pack(cp, rp, max_iter=4000)
            if s > best_sum:
                best_sum = s
                best_c = co
                best_r = ro

    # Ensure all radii are non-negative and centers are valid
    if best_c is not None:
        best_r = np.maximum(best_r, 1e-10)
        best_c = np.clip(best_c, best_r, 1 - best_r)

    return best_c, best_r, best_sum
import numpy as np
from report import _quadrant_ab_phases   # 你的模块名
from typing import Tuple

def phase_error(est: float, true: float) -> float:
    """Wrap phase difference to (−π, π]"""
    return np.mod(est - true + np.pi, 2*np.pi) - np.pi

def make_cosine_block(p: int, a: int, b: int, phi_a: float, phi_b: float) -> np.ndarray:
    """生成一个 p×p 网格：cos(2πa·y/p + φ_a) + cos(2πb·x/p + φ_b)"""
    y, x = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return (
        np.cos(2*np.pi*a*y/p + phi_a) +
        np.cos(2*np.pi*b*x/p + phi_b)
    )

def validate_phase_pipeline(p: int = 32, N: int = 100, verbose: bool = False):
    """随机生成 N 个 2p×2p 网格，验证四个象限的频率与相位提取是否准确"""
    rng = np.random.default_rng(42)
    all_ok = True

    for trial in range(N):
        specs = []
        quads = []
        for _ in range(4):
            a = rng.integers(1, p//2)
            b = rng.integers(1, p//2)
            phi_a = rng.random() * 2 * np.pi
            phi_b = rng.random() * 2 * np.pi
            specs.append(dict(a=a, b=b, phi_a=phi_a, phi_b=phi_b))
            quads.append(make_cosine_block(p, a, b, phi_a, phi_b))

        # 组合成 2p×2p grid
        grid = np.zeros((2*p, 2*p))
        grid[:p,  :p ] = quads[0]  # BL
        grid[:p,  p:]  = quads[1]  # BR
        grid[p:,  :p ] = quads[2]  # TL
        grid[p:,  p:]  = quads[3]  # TR

        # 相位提取
        estimates = _quadrant_ab_phases(grid)

        for i, ((phi_b_est, phi_a_est), spec) in enumerate(zip(estimates, specs)):
            db = phase_error(phi_b_est, spec["phi_b"])
            da = phase_error(phi_a_est, spec["phi_a"])

            if abs(db) > 1e-10 or abs(da) > 1e-10:
                all_ok = False
                print(f"[❌ FAIL] Trial {trial}, Q{i}: "
                      f"(b,a)=({spec['b']},{spec['a']}) | "
                      f"Δφ_b={db:+.2e}, Δφ_a={da:+.2e}")

            elif verbose:
                print(f"[OK] Trial {trial}, Q{i}: "
                      f"Δφ_b={db:+.2e}, Δφ_a={da:+.2e}")

            # Bonus: 重建信号检查
            y, x = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
            recon = (
                np.cos(2*np.pi*spec["a"]*y/p + phi_a_est) +
                np.cos(2*np.pi*spec["b"]*x/p + phi_b_est)
            )
            original = quads[i]
            if not np.allclose(recon, original, atol=1e-10):
                all_ok = False
                print(f"[❌ FAIL] Trial {trial}, Q{i}: signal reconstruction mismatch")

    if all_ok:
        print(f"\n✅ All {N*4} quadrant tests passed with ≤1e-10 phase error and perfect signal match.")
    else:
        print(f"\n❌ Some tests failed. Check above logs.")
def test_vertical_only():
    p = 32
    a = 4
    phi_a = 1.25  # rad
    b = 0         # no horizontal freq

    # 生成只含纵向正弦的 grid (2p × 2p, 各象限都相同)
    y, x = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    q = np.cos(2 * np.pi * a * y / p + phi_a)     # only vertical variation

    # 构造大网格：四个子块都是一样的纵向成分
    grid = np.zeros((2*p, 2*p))
    grid[:p, :p] = q       # BL
    grid[:p, p:] = q       # BR
    grid[p:, :p] = q       # TL
    grid[p:, p:] = q       # TR

    # 提取每个象限的 (φ_b, φ_a)
    results = _quadrant_ab_phases(grid)

    for i, (phi_b, phi_a_hat) in enumerate(results):
        db = np.mod(phi_b + np.pi, 2*np.pi) - np.pi
        da = np.mod(phi_a_hat - phi_a + np.pi, 2*np.pi) - np.pi
        print(f"Q{i}:  φ_b = {phi_b:.3f}, φ_a = {phi_a_hat:.3f}  |  "
              f"Δφ_b = {db:+.2e}, Δφ_a = {da:+.2e}")

    
# 🔧 Run test
if __name__ == "__main__":
    validate_phase_pipeline(p=32, N=100, verbose=False)
    test_vertical_only()
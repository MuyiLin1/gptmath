"""The isotropic minimizer rotates (Proposition "iso-rot").

D = beta*Id + eps*A with A = A_sparse (omega = +e3), V = -Id, on S^2.
Claim: at the uniform measure sigma the velocity field is exactly
    V_eps[sigma](x) = eps * rho(r)/r * A x,   r = sqrt(beta^2 + eps^2 |Ax|^2),   rho(r) = coth r - 1/r,
a rotation about omega whose angular rate is eps*rho(beta)/beta + O(eps^3).
(1) The field is evaluated by direct quadrature over the sphere and compared with the formula.
(2) The predicted mean angular rate over sigma is printed for the eps of the isotropic experiment
    (plot_isotropic.py, beta = 1), whose measured rates are 0.01565 and 0.03129.
Requires numpy only.
"""
import numpy as np

A = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
rho = lambda r: 1 / np.tanh(r) - 1 / r

# midpoint grid in (z, phi): uniform in area, so it integrates against sigma exactly up to O(1/n^2)
n = 1500
z = (np.arange(n) + 0.5) / n * 2 - 1; ph = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * np.pi
Zg, Pg = np.meshgrid(z, ph, indexing="ij"); Rg = np.sqrt(1 - Zg ** 2)
Y = np.stack([Rg * np.cos(Pg), Rg * np.sin(Pg), Zg], -1).reshape(-1, 3)

print("Part 1. Field at the uniform measure: direct quadrature vs formula")
rng = np.random.default_rng(1)
for beta in (0.5, 1.0, 2.0):
    for eps in (0.05, 0.1, 0.5):
        D = beta * np.eye(3) + eps * A; err = 0.0; perp = 0.0
        for x in rng.normal(size=(8, 3)):
            x /= np.linalg.norm(x)
            w = np.exp(Y @ (D.T @ x)); m = (w[:, None] * Y).sum(0) / w.sum()
            v = -(m - (x @ m) * x)
            Ax = A @ x; r = np.sqrt(beta ** 2 + eps ** 2 * (Ax @ Ax))
            pred = eps * rho(r) / r * Ax
            err = max(err, np.linalg.norm(v - pred) / np.linalg.norm(pred))
            u = Ax / np.linalg.norm(Ax); perp = max(perp, np.linalg.norm(v - (v @ u) * u) / np.linalg.norm(v))  # part not along the rotation
        print(f"  beta={beta:3.1f} eps={eps:4.2f}: max relative error {err:.1e}; max fraction of the field not along the rotation {perp:.1e}")

print("\nPart 2. Predicted spin rate about omega (beta = 1)")
z = np.linspace(-1, 1, 400001)
for eps, measured in ((0.05, 0.01565), (0.1, 0.03129)):
    r = np.sqrt(1 + eps ** 2 * (1 - z ** 2))
    mean_rate = np.trapezoid(eps * rho(r) / r, z) / 2
    print(f"  eps={eps:4.2f}: small-eps rate eps*rho(1) = {eps * rho(1.0):.5f}; exact mean over sigma = {mean_rate:.5f}; "
          f"measured in plot_isotropic.py = {measured:.5f}")

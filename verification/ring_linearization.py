"""A2 (clean): linearize the symmetric flow at the *idealized* uniform equatorial
ring and classify every mode analytically.

The clumped finite-N steady state of A1 carries convergence noise. Here we place
N tokens EXACTLY uniformly on the equator,

    X_k = (cos phi_k, sin phi_k, 0),   phi_k = 2*pi*k/N,

which is an exact stationary state of the symmetric field (the planar centroid is
0, so V_0 = 0). At this clean state the Jacobian has an exactly computable
structure, and we check the numerics against it.

ANALYTIC PREDICTION (V = Id, sgn = +1; beta drops out at linear order because the
kernel depends on z_k z_j, which is second order in any tangent displacement):

  * LATITUDE (out-of-plane) block  J_lat[k,j] = -1/N      (rank 1)
        eigenvalues: -1  once  (collective z-tilt of the whole ring)
                      0   (N-1) (zero-mean latitude ripples)
  * AZIMUTH (in-plane) block       J_azi[k,j] = -(1/N) cos(phi_k - phi_j)  (rank 2)
        eigenvalues: -1/2 twice (in-plane translation of the ring's centroid,
                                  the cos/sin modes)
                      0   (N-2) (azimuthal redistribution, incl. rigid rotation)

So the ONLY linearly contracting directions are the THREE rigid center-of-mass
motions that pin the circle as the z=0 great circle (one z-tilt at rate 1, two
in-plane translations at rate 1/2). The full (2N-3)-dimensional space of
on-ring redistributions and latitude ripples is NEUTRAL at linear order
(stabilized only nonlinearly / at higher order in beta). Rigid rotation about z
is one neutral vector in that space -- the free phase theta of prop:drift.

Run from repo root:  PYTHONPATH=. python exps/ring_linearization.py
"""
import os
import torch

torch.set_default_dtype(torch.float64)

D_DIM = 3
N = int(os.environ.get("PROBE_N", 48))
BETA = float(os.environ.get("PROBE_BETA", 1.0))
SGN = 1

Ds = torch.diag(torch.tensor([0.0, 0.0, 1.0]))


def proj_tang(x, y):
    return y - (y * x).sum(-1, keepdim=True) * x


def sym_velocity(x):
    K = torch.exp(BETA * (x @ (Ds @ x.T)))
    K = K / K.sum(dim=1, keepdim=True)
    update = K @ x
    return -SGN * proj_tang(x, update)


def uniform_ring(n):
    phi = 2 * torch.pi * torch.arange(n) / n
    x = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros(n)], dim=1)
    return x, phi


def main():
    x, phi = uniform_ring(N)
    resid = torch.linalg.vector_norm(sym_velocity(x), dim=-1).max().item()
    print(f"N = {N}, beta = {BETA}")
    print(f"uniform equatorial ring: max residual speed = {resid:.2e}  "
          f"(0 => exact stationary state)")

    def vfield_flat(xf):
        return sym_velocity(xf.reshape(N, D_DIM)).reshape(-1)

    J = torch.autograd.functional.jacobian(vfield_flat, x.reshape(-1))
    J = J.reshape(3 * N, 3 * N)

    # Per-token tangent frame: azimuth t = (-sin, cos, 0), latitude = z_hat.
    t_azi = torch.stack([-torch.sin(phi), torch.cos(phi), torch.zeros(N)], dim=1)
    t_lat = torch.zeros(N, 3)
    t_lat[:, 2] = 1.0

    T = torch.zeros(3 * N, 2 * N)
    for k in range(N):
        T[3 * k:3 * k + 3, 2 * k] = t_azi[k]
        T[3 * k:3 * k + 3, 2 * k + 1] = t_lat[k]
    J_red = T.T @ J @ T

    azi = torch.arange(0, 2 * N, 2)
    lat = torch.arange(1, 2 * N, 2)
    J_azi = J_red[azi][:, azi]
    J_lat = J_red[lat][:, lat]

    ev_lat = torch.linalg.eigvals(J_lat).real
    ev_azi = torch.linalg.eigvals(J_azi).real
    ev_lat = torch.sort(ev_lat).values
    ev_azi = torch.sort(ev_azi).values

    print("\n--- LATITUDE block eigenvalues (predicted: -1 once, 0 x (N-1)) ---")
    print(f"most negative : {ev_lat[0]:+.4f}   (predict -1.0)")
    print(f"2nd negative  : {ev_lat[1]:+.4f}   (predict  0.0)")
    print(f"# below -0.01 : {(ev_lat < -0.01).sum().item()}   (predict 1)")
    print(f"# near 0      : {(ev_lat.abs() < 1e-6).sum().item()} / {N}")

    print("\n--- AZIMUTH block eigenvalues (predicted: -1/2 twice, 0 x (N-2)) ---")
    print(f"most negative : {ev_azi[0]:+.4f}   (predict -0.5)")
    print(f"2nd negative  : {ev_azi[1]:+.4f}   (predict -0.5)")
    print(f"3rd negative  : {ev_azi[2]:+.4f}   (predict  0.0)")
    print(f"# below -0.01 : {(ev_azi < -0.01).sum().item()}   (predict 2)")
    print(f"# near 0      : {(ev_azi.abs() < 1e-6).sum().item()} / {N}")

    # Cross-check: rigid rotation about z is a null vector.
    xi = torch.zeros(N, 3)
    xi[:, 0] = -x[:, 1]
    xi[:, 1] = x[:, 0]
    xi = xi.reshape(-1)
    xi = xi / torch.linalg.vector_norm(xi)
    print("\n--- rigid z-rotation generator ---")
    print(f"||J xi|| = {torch.linalg.vector_norm(J @ xi):.2e}   (predict 0)")

    print("\n--- SUMMARY ---")
    print("Only the 3 rigid placements of the circle contract at linear order:")
    print(f"  z-tilt           rate ~ {-ev_lat[0]:.3f}")
    print(f"  in-plane shift x2 rate ~ {-ev_azi[0]:.3f}, {-ev_azi[1]:.3f}")
    print(f"Remaining {2*N-3} on-ring / latitude-ripple modes are linearly neutral.")
    print("Normal hyperbolicity is therefore in the 3 rigid modes that pin the")
    print("z=0 great circle; on-ring redistribution (incl. the rotation phase) is")
    print("neutral at linear order -- the tie the O(eps) ghost force breaks.")


if __name__ == "__main__":
    main()

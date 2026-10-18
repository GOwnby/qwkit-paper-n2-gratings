import numpy as np
from qwkit import interface as I


def test_parabolic_subband_density():
    D = I.dos_factor(0.2)
    for mu, T in ((0.1, 10.0), (0.05, 300.0), (-0.05, 300.0)):
        exact = D * I.KB * T * np.logaddexp(0, mu / (I.KB * T))
        assert abs(I.subband_density(0.0, 0.2, 1e9, mu, T, ne=4000) / exact - 1) < 2e-4


def test_nonparabolic_density_degenerate():
    # T -> 0: n = D (eF + eF^2/G)
    D = I.dos_factor(0.2); eF = 0.3; G = 1.5
    n = I.subband_density(0.0, 0.2, G, eF, 1.0, ne=20000)
    assert abs(n / (D * (eF + eF ** 2 / G)) - 1) < 1e-3


def test_self_consistent_neutral_and_sum():
    r = I.solve(2e13, T=300.0, NA=1e19, E_pin=1.0, nstates=6, L=300.0)
    assert abs(r["n_n"].sum() / 2e13 - 1) < 1e-6
    # the potential is flat beyond the depletion edge (neutral)
    z, V = r["z"], r["V"]
    far = z > r["w_d"] + 20
    assert np.ptp(V[far]) < 2e-3
    # pinning condition
    assert abs(V[-1] - r["mu"] - 1.0) < 5e-3


def test_xc_potential_value():
    # rs = 1 in effective units: HL gives -(2/(pi alpha))(1 + 0.7734 x ln(1+1/x)) Ry, x = 1/21
    eps, m = 10.0, 0.1
    a = 0.529177e-8 * eps / m
    n = 3 / (4 * np.pi * a ** 3)
    alpha = (4 / (9 * np.pi)) ** (1 / 3); x = 1 / 21
    ref = -(2 / (np.pi * alpha)) * (1 + 0.7734 * x * np.log(1 + 1 / x)) * 13.605693 * m / eps ** 2
    assert abs(I.xc_potential(np.array([n]), eps, m)[0] / ref - 1) < 1e-9


def test_dispersion_inverse():
    G = 2.0; m = 0.2
    eps = np.array([0.0, 0.1, 0.5])
    x = eps * (1 + eps / G)
    assert np.allclose(I.dispersion(x, m, G), eps)


def test_cap_donors_gauss():
    """Ionized donors in the cap change the cap field but, by Gauss's law, not the well side: with the same electron
    sheet and the rest of the positive charge at the interface, the subbands are unchanged unless electrons enter the
    cap."""
    import numpy as np
    from qwkit import interface as I
    kw = dict(T=300.0, NA=5e19, L=300.0, dz=0.5, nstates=4)
    r0 = I.solve(3e13, **kw)
    r1 = I.solve(3e13, nd_cap=lambda z: np.full_like(z, 2e19), **kw)
    assert abs(r1["N_D"] - 1e13) < 1e10                  # 5 nm x 2e19 cm^-3
    assert abs((r1["E"][1] - r1["E"][0]) - (r0["E"][1] - r0["E"][0])) < 2e-3
    cap = r1["z"] < 0
    assert np.min(r1["V"][cap]) < np.min(r0["V"][cap]) - 0.05   # donors bend the cap band down towards the surface


def test_rest_to_donors_and_interface():
    """rest_to='donors' scales the donor shape to carry all positive charge beyond sigma_int (including the change of
    the depletion charge); rest_to='interface' puts everything not in fixed donors at the interface. Both neutral."""
    import numpy as np
    from qwkit import interface as I
    kw = dict(T=300.0, NA=5e19, L=300.0, dz=0.5, nstates=4)
    sP = 3e13
    d = I.solve(6e13, sigma_int=sP, nd_cap=lambda z: np.where(np.abs(z + 10.0) <= 1.0, 1.0, 0.0), rest_to="donors", **kw)
    assert abs(d["sig_cap"]) < 1 and abs(d["sig_int"] + d["N_D"] - 6e13 - d["N_dep"]) < 1e6
    i = I.solve(6e13, nd_cap=lambda z: np.full_like(z, 1e19), rest_to="interface", **kw)
    assert abs(i["N_D"] - 5e12) < 1e6 and i["sig_cap"] == 0 and abs(i["sig_int"] + i["N_D"] - 6e13 - i["N_dep"]) < 1e6

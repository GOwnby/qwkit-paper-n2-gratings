"""Further analytic checks: Fermi-Dirac subband populations, the magnitude of the depolarisation shift,
Gauss's law across a dielectric interface, the refractive index used in absorption cross-sections, and
modal-overlap normalisation."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwkit import core as q, sp, propagation as pr, waveguide as wg

HBAR = 1.054571817e-34; M0 = 9.1093837e-31; KB = 8.617333e-5


def test_populations_closed_form():
    # 2D Fermi-Dirac: n_i = D kT ln(1 + exp((mu - E_i)/kT)), D = m/(pi hbar^2); with one subband populated the
    # chemical potential is mu = E_1 + kT ln(exp(ns/(D kT)) - 1).
    E = np.array([0.0, 0.5, 1.0]); ns, T, m = 5e11, 300.0, 0.043
    rho = q.populations(E, ns, T, m_eff=m)
    D = m * M0 / (np.pi * HBAR ** 2) * q.E_CH * 1e-4
    kT = KB * T
    mu = kT * np.log(np.expm1(ns / (D * kT)))
    assert abs(rho.sum() - 1) < 1e-12
    assert abs(rho[1] - D * kT * np.log1p(np.exp((mu - 0.5) / kT)) / ns) < 1e-12
    assert abs(rho[0] - 1.0) < 1e-6


def test_depolarisation_shift_infinite_well():
    # Infinite well, ground state fully occupied: S_12 = int [int^z psi1 psi2]^2 = 5 L / (9 pi^2), and
    # E~^2 = E^2 (1 + 2 e^2 ns S / (eps eps0 E)).
    L, dz, ns, eps = 100.0, 0.05, 3e11, 13.5
    z = (np.arange(int(L / dz)) + 0.5) * dz
    psi = np.sqrt(2 / L) * np.column_stack([np.sin(np.pi * z / L), np.sin(2 * np.pi * z / L)])
    E = np.array([0.05, 0.20]); rho = np.array([1.0, 0.0])
    Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=eps)
    S = 5 * L * 1e-10 / (9 * np.pi ** 2)
    E21 = 0.15
    expect = E21 * np.sqrt(1 + 2 * q.E_CH ** 2 * ns * 1e4 * S / (eps * q.EPS0 * E21 * q.E_CH))
    assert abs(Et[1, 0] - expect) < 1e-4 * (expect - E21)


def test_gauss_law_across_dielectric_interface():
    # A +sheet / -sheet pair with two dielectrics: D is piecewise constant and continuous at the interface,
    # so the field jumps by eps1/eps2 there; the field has zero mean over the period.
    L, dz, ns = 400.0, 0.5, 5e11
    z = (np.arange(int(L / dz)) + 0.5) * dz
    nd = np.zeros_like(z); n = np.zeros_like(z)
    nd[np.argmin(abs(z - 50))] = ns / dz; n[np.argmin(abs(z - 350))] = ns / dz
    eps = np.where(z < 200, 13.9, 12.4)
    V = sp.hartree(z, nd, n, eps)
    Efield = -np.gradient(-V, z)                     # electron potential energy -> field (arb. sign)
    i1, i2 = np.argmin(abs(z - 190)), np.argmin(abs(z - 210))
    assert abs(Efield[i1] * 13.9 - Efield[i2] * 12.4) < 1e-6 * abs(Efield[i1] * 13.9)
    assert abs(np.mean(Efield)) < 1e-9 * np.max(abs(Efield))


def test_cross_section_scales_with_refractive_index():
    hw = 0.124
    st = dict(E=[0.0, hw], Z=[[0.0, 20.0], [20.0, 0.0]], Et=[[0.0, -hw], [hw, 0.0]], rho=[1.0, 0.0], N=1e17)
    a = pr.Medium(st, hw, nref=3.3).sig1[0, 1]; b = pr.Medium(st, hw, nref=1.0).sig1[0, 1]
    assert abs(b / a - 3.3) < 1e-12


def test_overlap_normalisation():
    # Modes are normalised to int(|Ez|^2 + |Ex|^2) dz = 1, so the Ez confinement over the whole guide equals
    # 1 minus the Ex share, which is small for a weakly guiding TM mode.
    layers = [dict(material="InP", t_um=15.0), dict(material="InGaAs", t_um=3.0), dict(material="InP", t_um=15.0)]
    m = wg.tm_modes(layers, 10.0, dz_um=0.05, nmodes=1)[0]
    dz = m["z"][1] - m["z"][0]
    ex_share = float(np.sum(np.abs(m["Ex"]) ** 2) * dz)
    total = wg.overlap(m, m["z"][0] - 1.0, m["z"][-1] + 1.0)
    core_only = wg.overlap(m, 15.0, 18.0)
    assert abs(total + ex_share - 1.0) < 1e-9 and ex_share < 0.1 and 0.0 < core_only < total

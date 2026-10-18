"""qwkit.thermal: material data, Planck flux, coupled-mode emissivity, detailed balance and ISB absorption limits."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from qwkit import thermal as th


def test_gaps_and_intrinsic_density():
    assert abs(th.gap("GaAs", 300.0) - 1.422) < 2e-3          # Vurgaftman et al. (2001)
    assert abs(th.gap("InGaAs", 300.0) - 0.735) < 2e-3
    ni = th.intrinsic("GaAs", 300.0)
    assert 1e6 < ni < 5e6                                      # ~2e6 cm^-3


def test_planck_flux_integrates_to_stefan_boltzmann():
    hw = np.linspace(0.002, 3.0, 40000)
    for T in (500.0, 1000.0):
        P = np.trapezoid(th.planck_flux(hw, T) * hw * th.QE, hw)
        assert abs(P / (5.670374419e-8 * T ** 4) - 1) < 1e-4


def test_critical_coupling_gives_unit_emissivity():
    hw = np.linspace(0.2, 0.3, 2001)
    g = th.alpha_to_rate(1000.0, 0.3)
    Qr = 0.25 / (2 * g)
    e, res, _ = th.emitter_spectrum(hw, np.full_like(hw, 1000.0), 600.0, 0.25, Qr, confinement=0.3)
    assert abs(res.max() - 1) < 1e-3


def test_tpv_detailed_balance_limits():
    hw = np.linspace(0.005, 3.0, 20000)
    bb = th.tpv(hw, np.ones_like(hw), 1000.0, Eg=0.22, sub_reflect=0.95)
    assert 0 < bb["eta"] < 1 - 300.0 / 1000.0                  # below Carnot
    narrow = ((hw >= 0.22) & (hw <= 0.26)).astype(float)
    bp = th.tpv(hw, narrow, 1000.0, Eg=0.22, sub_reflect=0.95)
    assert bp["eta"] > bb["eta"] and bp["P_max"] < bb["P_max"]  # narrowband: more efficient, less power


def test_isb_alpha_parabolic_limit_conserves_strength():
    """With equal in-plane masses the k-resolved line is a single Lorentzian whose area is temperature independent
    for a two-level system with only the ground subband occupied."""
    E = np.array([0.0, 0.25]); Z = np.array([[0.0, 15.0], [15.0, 0.0]])
    hw = np.linspace(0.05, 0.6, 5501)
    areas = []
    for T in (300.0, 600.0):
        a = th.isb_alpha(hw, E, Z, None, np.array([0.05, 0.05]), 5e11, 250.0, T, 0.005)
        areas.append(np.trapezoid(a / hw, hw))
    assert abs(areas[1] / areas[0] - 1) < 2e-2

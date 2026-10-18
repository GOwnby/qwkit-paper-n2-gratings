"""Validation of the TM slab solver against the analytic symmetric-slab dispersion relation."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import brentq
from qwkit import waveguide as wg

def test_symmetric_slab_tm0():
    lam, d, n1, n2 = 10.0, 2.0, 3.4, 3.1
    k0 = 2 * np.pi / lam
    def f(ne):
        kap = k0 * np.sqrt(n1 ** 2 - ne ** 2); gam = k0 * np.sqrt(ne ** 2 - n2 ** 2)
        return np.tan(kap * d / 2) - (n1 ** 2 / n2 ** 2) * gam / kap
    ne_th = brentq(f, n2 + 1e-6, n1 - 1e-6)
    layers = [dict(material="X1", t_um=8.0), dict(material="X2", t_um=d), dict(material="X1", t_um=8.0)]
    orig = wg.layer_eps
    wg.layer_eps = lambda L, lam_um: {"X1": n2 ** 2 + 0j, "X2": n1 ** 2 + 0j}[L["material"]]
    try:
        m = wg.tm_modes(layers, lam, dz_um=0.01, nmodes=1)[0]
    finally:
        wg.layer_eps = orig
    assert abs(m["neff"].real - ne_th) < 2e-4, (m["neff"], ne_th)

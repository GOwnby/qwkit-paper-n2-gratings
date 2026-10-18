"""qwkit: quantum-well intersubband optics toolkit.

Modules
-------
core         one-band envelope-function states, dipoles, populations, density-matrix chi(2)
sp           self-consistent Schroedinger-Poisson for a voltage-biased doped period, depolarisation
scattering   Froehlich LO-phonon intersubband rates with detailed balance
waveguide    TM modes of planar multilayer waveguides, confinement and nonlinear overlap
electrodes   fields under alternating-polarity electrodes and the QPM Fourier coefficient
propagation  saturable multilevel media and coupled-mode SHG with quasi-phase matching
tdh          collective (time-dependent Hartree) linear and second-order subband response, real time
krpa         k-resolved RPA/TDH response with in-plane dispersion and a polar-phonon background
sens         exact parameter sensitivities by implicit differentiation (matrix-free), UQ helpers
interface    self-consistent 2DEG at a single polar heterointerface (Kane in-plane dispersion, pinning)
thermal      temperature-dependent intersubband thermal emitters, leakage and TPV estimates
"""
from . import core, sp, scattering, waveguide, electrodes, propagation, tdh, krpa, sens, interface, thermal, transport
from .core import add_material

__version__ = "0.7.0"
__all__ = ["core", "sp", "scattering", "waveguide", "electrodes", "propagation", "tdh", "krpa", "sens", "interface", "thermal", "add_material", "__version__"]

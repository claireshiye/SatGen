#!/usr/bin/env python
"""
run_minor_mergers.py -- evolve TNG100 minor-merger satellites with SatGen,
following the recipe of test_evolve.py, with every host, satellite and
orbit parameter read from TNG100_minor_mergers_satgen.csv instead of being
typed in by hand.

One CSV row = one (host, dwarf) pair at snapshot `snapnum` (redshift `z`).

    CSV column(s)                            -> SatGen input (test_evolve name)
    -------------------------------------------------------------------------
    HOST (fitted, static, evaluated at the row's z)
    host_halo_kind, host_Mh_Msun, host_ch,
        host_alpha                           -> Einasto(Mh, c, alpha, z)      (hDekel)
    host_Md_Msun, host_a_kpc, host_b_kpc     -> MN(Md, a, b)                  (dMN)
    host_Mb_Msun, host_ab_kpc                -> Hernquist bulge
    host_Me_Msun, host_ae_kpc                -> Hernquist envelope
    host_Msph_Msun, host_asph_kpc            -> Hernquist spheroid (no disk)
    host_source_file, prikey, host_subkey    -> TabulatedGas corona (default): the
                                                measured rho_1d_hot of the host, read
                                                from TNG_profiles/<host_source_file>,
                                                log-interpolated, flat inside the first
                                                bin, power-law beyond the aperture
                                                -> gpotential                 (gDekel)
    fitted halo x M_hot/M_halo (measured)    -> halo-shaped corona, with --hot-gas
                                                scaled: test_evolve's gDekel with
                                                fgh measured per host
    fitted halo x --fgh                      -> test_evolve's gDekel, --hot-gas fgh
    host_hot_rho0, host_hot_rc_kpc,
        host_hot_beta                        -> BetaGas corona, with --hot-gas beta
    (an unbuildable measured corona means NO corona; the summary says why)
    (same HDF5 host)                         -> TabulatedGas2D ISM (default): the
                                                measured cold-gas (R,|z|) grid,
                                                inpainted inside its envelope,
                                                zero outside it -> gpotential  (gMN)
    host_M*/a* stellar model x M_cold/M_star -> ScaledGas ISM, with --cold-gas
                                                scaled (explicit choice only; an
                                                unbuildable grid means no ISM)
    --fgd x stellar model                    -> test_evolve's gMN, with --cold-gas fgd
    --fgd x stellar model                    -> cold ISM        -> gpotential  (gMN)

    SATELLITE (structure at the dwarf's peak-mass snapshot dw_max_snapnum)
    max_Mtot_dw_Msun                         -> mv0  (virial mass at infall)
    max_Rhm_tot_dw_kpc                       -> c0   (Dekel c with M(<Rhm)=mv0/2)
    --alpha-sat (default cfg.eps)            -> aDekel0
    max_Mgas_dw_Msun / max_Mtot_dw_Msun      -> fg0
    max_M_dw_Msun                            -> ms0
    max_Rhm_dw_kpc                           -> leff0 (3D half-stellar-mass radius)
    M_dw_tot_Msun, M_dw_gas_Msun             -> starting masses when --msat current
                                                (pre-stripped along the P10 track)

    ORBIT (at snapnum)
    D_kpc                                    -> r0 = sqrt(R0^2 + z0^2)
    Vrel_kms * cfg.kms2kpcGyr                -> |V0|
    SMA_kpc, ECC                             -> V_tan/|V| = a sqrt((1-e^2)/(D(2a-D)))
                                                (exact for the two-body elements
                                                 the notebook computed from D, V)
    --inc / --vr-sign                        -> orientation w.r.t. the disk and
                                                approaching/receding (not in CSV)

    TIME
    z                                        -> tmax = lookback time to z=0 (Gyr)
    --nstep                                  -> Nstep

    REDSHIFTS (--snap-z table, snapnum -> z)
    z(snapnum + snap_offset)                 -> host construction (the fit's z;
                                                rh is checked against host_rh_kpc)
    z(dw_max_snapnum)                        -> satellite c0 solve (where Rhm was
                                                measured); the profile is then
                                                re-expressed at the row's z with
                                                ev.Dekel, which keeps it physically
                                                identical

SatGen issues worked around in THIS FILE (the SatGen library is deliberately
left unchanged; nothing here is monkey-patched into it):

    1. Hernquist velocity dispersion returns 0 far out -> NaN dynamical
       friction.  See class HernquistSafe.  Every Hernquist component built
       by build_host uses that subclass.  Code that rebuilds hosts through
       the library instead (satgen_host_fitter.components_from_params,
       revalidate.py, df_check.py, a hand-written Hernquist(Mb, ab)) is NOT
       protected.
    (Both gas terms of `gpotential` are taken from the extraction, not from
    fits, by default: see load_row_host, --hot-gas and --cold-gas.)

    1b. Einasto velocity dispersion returns NaN where the density has
       underflowed to 0 (r >~ 30-40 rh for fits with alpha ~ 1) -> NaN
       friction -> stalled integrator.  See class EinastoSafe.

    2. evolve.lram reports "all gas stripped" whenever the restoring force is
       below the ram pressure at BOTH bracket ends, which is always the case
       for a cored satellite gas profile.  See lram_robust / mgas_robust;
       --upstream-lram restores the library behaviour.
    3. orbit.integrate never checks scipy's solver.successful(); a stalled
       dopri5 leaves the orbit frozen while stripping continues.  The loop in
       evolve_pair compares o.t with the requested time and raises instead.

Usage (from the SatGen directory):

    python run_minor_mergers.py ~/Documents/Research/Off-nuclearTDE/TNG100_minor_mergers_satgen.csv \
        --limit 20 --quality-ok --bound-only -o fits/minor_mergers_summary.csv -v

    python run_minor_mergers.py FILE.csv \
        --select prikey=0,host_subkey=10000000000000001,dw_subfindid=822 \
        --save-tracks tracks/ -v

    python run_minor_mergers.py FILE.csv --dry-run --limit 5     # inputs only, no evolution
"""

# ---user modules (SatGen)
import config as cfg
import cosmo as co
from profiles import NFW, Dekel, Einasto, MN, Vcirc, tdyn
import profiles as pr
from orbit import orbit
import evolve as ev
import galhalo as gh
# patched Hernquist (.fgrav/.sigma), the gas-density callables and the HDF5
# loader used by the fitter
from satgen_host_fitter import (Hernquist, BetaGas, TabulatedGas, TabulatedGas2D,
                                ScaledGas, load_host, tabulated_gas,
                                tabulated_cold_gas, scaled_cold_gas)

# ---python modules
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import argparse
import os
import sys
import time

SNAP_Z_DEFAULT = os.path.expanduser(
    '~/Documents/Research/Off-nuclearTDE/redshift_snapnum.txt')


def load_snap_z(path):
    """snapnum -> redshift dict from a two-column text table; None if absent."""
    if path and os.path.exists(path):
        t = np.loadtxt(path)
        return dict(zip(t[:, 0].astype(int), t[:, 1]))
    return None


def _z_of(snap_z, snap, fallback):
    return float(snap_z[int(snap)]) if snap_z is not None else float(fallback)


# ======================================================================
# host hot gas from the extraction files
# ======================================================================

HOT_GAS_CHOICES = ('tabulated', 'scaled', 'fgh', 'beta', 'none')
COLD_GAS_CHOICES = ('tabulated', 'scaled', 'fgd', 'none')
PROFILES_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TNG_profiles')

_H5_CACHE = {}


def _open_h5(path):
    """
    Per-process cache of open HDF5 files.  Opened lazily on first use so that
    with --nproc > 1 every fork()ed worker opens its own handle: an HDF5 file
    handle inherited across fork() is not safe to use.
    """
    f = _H5_CACHE.get(path)
    if f is None:
        import h5py
        f = h5py.File(path, 'r')
        _H5_CACHE[path] = f
    return f


def load_row_host(row, profiles_dir):
    """
    The fitter's host dictionary for this row, read from the extraction file
    named in host_source_file (keys in the HDF5 are strings).  It feeds the
    two measured gas terms of `gpotential`:

        tabulated_gas(host)       spherical hot-gas density (T >= 1e5 K, SFR=0)
        tabulated_cold_gas(host)  cold-gas (R,|z|) grid, the ISM
        scaled_cold_gas(host, b)  fitted stellar shape x measured M_cold/M_star

    exactly what satgen_host_fitter.build_host hands to evolve.mgas.  Only
    .rho() of these objects is ever used (through profiles.rho in
    evolve.Findlrp); both gas masses are already inside the fitted halo, so
    none of them must be added to `potential`.
    """
    path = os.path.join(profiles_dir, str(row['host_source_file']))
    return load_host(_open_h5(path), str(int(row['prikey'])), str(int(row['host_subkey'])))


def _baryons_from_row(row):
    """The stellar-model dict scaled_cold_gas expects, rebuilt from the CSV columns."""
    if row['host_stellar_model'] == 'spheroid':
        return {'kind': 'spheroid', 'M': float(row['host_Msph_Msun']), 'a': float(row['host_asph_kpc'])}
    b = {'kind': 'disk', 'Md': float(row['host_Md_Msun']), 'a': float(row['host_a_kpc']),
         'b': float(row['host_b_kpc'])}
    if _ok(row['host_Mb_Msun']):
        b.update(Mb=float(row['host_Mb_Msun']), ab=float(row['host_ab_kpc'])); b['kind'] += '+bulge'
    if _ok(row['host_Me_Msun']):
        b.update(Me=float(row['host_Me_Msun']), ae=float(row['host_ae_kpc'])); b['kind'] += '+envelope'
    return b


# ======================================================================
# host
# ======================================================================

class HernquistSafe(Hernquist):
    """
    satgen_host_fitter.Hernquist with a numerically stable .sigma().

    WHAT IS WRONG IN THE LIBRARY
    ----------------------------
    satgen_host_fitter.Hernquist.sigma (and the patched profiles.Hernquist
    .sigma, which is the same code) evaluates the exact isotropic dispersion

        sigma^2 = GM/(12a) [ 12x(1+x)^3 ln(1+1/x) - x/(1+x)(25+52x+42x^2+12x^3) ]

    with x = r/a.  The formula is correct, but it is the difference of two
    terms that each grow like 12 x^4 while their difference shrinks like
    (12/5)/x.  In float64 the two terms are stored to ~16 significant digits,
    so once 12x^4 / (2.4/x) ~ 5 x^5 approaches 1e16 the difference is pure
    rounding noise.  Measured against a 50-digit evaluation of the same
    formula (2026-10-19):

        x = r/a      error in sigma^2
          50         2e-7
         333         1e-3
        1000         order unity, random sign from one radius to the next
       >1100         first exact zero; beyond ~1500 about half of all
                     evaluations are negative and get clipped to sigma = 0

    Example at x = 1667 (r = 500 kpc, a = 0.3 kpc): the two terms are
    55638925558.549 and 55638925558.554, the true difference is +0.00144,
    the float64 difference is -0.00468 -> clipped -> sigma = 0, whereas the
    model's real sigma there is 7.3 kpc/Gyr.

    WHY A ZERO IS FATAL
    -------------------
    profiles.fDF loops over the host components and, for each one, forms
    X = V/(sqrt(2) sigma) and F(X) = erf(X) - (2/sqrt(pi)) X exp(-X^2).
    sigma = 0 gives X = inf and inf * exp(-inf) = inf * 0 = NaN, so F = NaN
    for that component.  The components' forces are then ADDED, and any sum
    containing a NaN is NaN: the halo's and disk's (correct, dominant)
    friction is destroyed by a bulge whose own contribution is ~1e-7 of the
    total.  dopri5 then stalls on the NaN acceleration (see the o.t check in
    evolve_pair).  Note the physics points the other way: a cold background
    (sigma << V) means F -> 1, the MAXIMUM friction per unit density; the
    far-away bulge is negligible only because rho_bulge ~ r^-4, not because
    of sigma.

    WHO IS AFFECTED
    ---------------
    Any orbit that reaches r > ~1100 a_min, with a_min the smallest Hernquist
    scale radius of the host (bulge ab, envelope ae, or spheroid asph).  In
    TNG100_minor_mergers_satgen.csv (160,152 rows) that is 54 rows already
    at their starting separation, ~560 bound rows at their Kepler apocentre
    (0.36%), plus an unknown share of the 4,734 unbound rows.  Almost all
    have a_min railed at the fitter's lower bound (0.05-0.2 kpc).

    WHAT THIS CLASS DOES
    --------------------
    For x <= X_SWITCH the closed form is used unchanged (bit-for-bit the
    library result).  For x > X_SWITCH the SAME quantity is evaluated with
    its large-x expansion, obtained by cancelling the 12x^3+30x^2+22x+3
    pieces of the two terms symbolically:

        sigma^2 = GM/a [ 1/(5x) - 7/(30x^2) + 17/(70x^3) - 69/(280x^4) + O(x^-5) ]

    X_SWITCH = 50 is where the two error budgets cross: the truncated series
    is off by ~(1/x)^4 = 1.6e-7 there and the closed form's rounding error is
    also ~1.6e-7; the switch could sit anywhere in 20-300 without changing
    sigma by more than 1e-6.  A simpler NaN patch would be to floor sigma at a
    tiny positive value: that gives the identical friction force (F is
    already 1 to double precision wherever the closed form fails) but leaves
    .sigma() returning a number that is wrong by orders of magnitude; the
    series returns the right one at the same cost.

    SCOPE
    -----
    The library classes are intentionally left as they are (user decision,
    2026-10-20).  This subclass protects only the hosts built by build_host
    in this file.  satgen_host_fitter.components_from_params, revalidate.py,
    df_check.py and any hand-built Hernquist(M, a) handed to orbit() with
    dynamical friction on still carry the bug.
    """
    X_SWITCH = 50.

    def sigma(self, R, z=0.):
        r = np.sqrt(np.asarray(R, dtype=float) ** 2 + np.asarray(z, dtype=float) ** 2)
        x = np.maximum(r / self.a, 1e-12)
        u = 1. / x
        s2_series = self.GMb / self.a * u * (0.2 + u * (-7. / 30. + u * (17. / 70. - u * 69. / 280.)))
        xs = np.minimum(x, self.X_SWITCH)             # keep the exact branch finite
        s2_exact = self.GMb / (12. * self.a) * (
            12. * xs * (1. + xs) ** 3 * np.log(1. + 1. / xs)
            - (xs / (1. + xs)) * (25. + 52. * xs + 42. * xs ** 2 + 12. * xs ** 3))
        s2 = np.where(x > self.X_SWITCH, s2_series, s2_exact)
        return np.sqrt(np.maximum(s2, 0.))


class EinastoSafe(Einasto):
    """
    profiles.Einasto with a finite .sigma() where the density has underflowed.

    WHAT IS WRONG IN THE LIBRARY
    ----------------------------
    The Einasto density is rho_s exp{-d(n)[(r/rs)^alpha - 1]}.  For the
    dwarf hosts whose fit railed at alpha ~ 1 (the upper bound) this is an
    exponential cut-off, and in float64 rho(r) is exactly 0 beyond r ~ 30-40
    rh (~600 kpc for rh ~ 18 kpc).  profiles.Einasto.sigma then divides 0 by
    0 and returns NaN, and profiles.fDF turns that into a NaN acceleration
    (see HernquistSafe for the same mechanism), which stalls dopri5.  Hit by
    satellites that have escaped a static dwarf host: rows 296 and 471 of the
    catalogue stalled at 721 and 582 kpc (42 and 31 rh).

    WHAT THIS CLASS DOES
    --------------------
    Where the library sigma is not finite and positive it returns a small
    positive placeholder.  This is a pure NaN guard, not a value: at those
    radii rho is exactly 0, so the friction from this component is 0 for
    ANY finite sigma, and the placeholder never enters a physical quantity.
    Inside the radius where rho underflows the library result is returned
    unchanged.  Library classes are left as they are.
    """
    SIGMA_PLACEHOLDER = 1e-10

    def sigma(self, R, z=0.):
        with np.errstate(all='ignore'):
            s = np.asarray(super().sigma(R, z), dtype=float)
        bad = ~np.isfinite(s) | (s <= 0.)
        if np.any(bad):
            s = np.where(bad, self.SIGMA_PLACEHOLDER, s)
        return s.item() if s.ndim == 0 else s


_HALO = {
    'NFW':     lambda Mh, c, a, z: NFW(Mh, c, Delta=200., z=z),
    'Dekel':   lambda Mh, c, a, z: Dekel(Mh, c, a, Delta=200., z=z),
    'Einasto': lambda Mh, c, a, z: EinastoSafe(Mh, c, a, Delta=200., z=z),
}


_HALO_SCALED = {   # same rh and rs as the halo, density x f  (test_evolve: gDekel)
    'NFW':     lambda Mh, c, a, z, f: NFW(Mh * f, c, Delta=200. * f, z=z),
    'Dekel':   lambda Mh, c, a, z, f: Dekel(Mh * f, c, a, Delta=200. * f, z=z),
    'Einasto': lambda Mh, c, a, z, f: Einasto(Mh * f, c, a, Delta=200. * f, z=z),
}


def scaled_halo_gas(row, z_host, f):
    """
    A copy of the fitted halo with its mass AND Delta multiplied by f: rh and
    rs are unchanged and the density is f times the halo's everywhere.  This
    is test_evolve.py's
        gDekel = Dekel(Mv*fgh, cDekel, aDekel, Delta=hDekel.Deltah*fgh)
    for whichever halo kind was fitted.  Used only through .rho() as a
    corona for evolve.mgas; NOT a member of `potential`.

    Caveat (measured 2026-10-21 on 300 hosts): the TNG hot gas is cored while
    the fitted halo is cuspy, so this shape overestimates the measured corona
    by ~14x (median) at 5 kpc and ~3x at 20 kpc, matches within ~10% at
    50-100 kpc, and is unrelated to it beyond the aperture.  Offered for
    comparison with test_evolve, not as a default.
    """
    gas = _HALO_SCALED[row['host_halo_kind']](row['host_Mh_Msun'], row['host_ch'],
                                              row['host_alpha'], z_host, f)
    if abs(gas.rh / row['host_rh_kpc'] - 1.) > 1e-6:
        raise ValueError('scaled halo changed rh: %.4f vs %.4f' % (gas.rh, row['host_rh_kpc']))
    return gas


def _ok(x):
    """True for a finite, positive CSV entry (NaN marks an absent component)."""
    try:
        return np.isfinite(x) and x > 0.
    except TypeError:
        return False


def build_host(row, snap_z=None, fgd=0., fgh=0., hot_gas='tabulated', cold_gas='tabulated',
               profiles_dir=None, rh_tol=0.05):
    """
    Rebuild the fitted host from one CSV row.

    The host profile was fitted at snapshot snapnum + snap_offset, so its
    (Mh, c) must be instantiated at THAT redshift to reproduce the fitted
    physical profile; the rebuilt rh is checked against host_rh_kpc.

    hot_gas selects the corona handed to evolve.mgas:
        'tabulated' (default) the measured profile from the HDF5 extraction
                    (needs profiles_dir); when it cannot be built (fewer than
                    3 bins with hot gas: hosts without a corona) there is NO
                    corona, with the reason in info['hot_gas']
        'scaled'    the fitted halo shape x the measured M_hot/M_halo inside
                    the fit aperture (needs profiles_dir); see scaled_halo_gas
        'fgh'       the fitted halo shape x the constant `fgh` (test_evolve)
        'beta'      the 3-parameter fit stored in the CSV (host_hot_* columns)
        'none'      no corona

    cold_gas selects the ISM term:
        'tabulated' (default) the measured cold-gas (R,|z|) grid; when it
                    cannot be built (empty or too sparse grid: gas-free
                    hosts) there is NO ISM, with the reason in
                    info['cold_gas'] -- see the comment in the code for why
                    the scaled model is not used as the fallback
        'scaled'    the fitted stellar model times the measured M_cold/M_star
                    inside the fit aperture (explicit choice only; falls back
                    to 'none' if it cannot be built)
        'fgd'       test_evolve's recipe: the stellar model times the constant
                    --fgd
        'none'      no ISM

    Returns (potential, gpotential, info) where
        potential:  list of profile objects -> orbit(), ev.msub()
        gpotential: list of gas-density objects -> ev.mgas()  (may be empty)
        info:       dict(labels, z_host, rh_mismatch, hot_gas, cold_gas)
    """
    if hot_gas not in HOT_GAS_CHOICES:
        raise ValueError('hot_gas must be one of %s' % (HOT_GAS_CHOICES,))
    if cold_gas not in COLD_GAS_CHOICES:
        raise ValueError('cold_gas must be one of %s' % (COLD_GAS_CHOICES,))
    z_host = _z_of(snap_z, row['snapnum'] + row['snap_offset'], row['z'])
    halo = _HALO[row['host_halo_kind']](row['host_Mh_Msun'], row['host_ch'],
                                        row['host_alpha'], z_host)
    rh_mismatch = halo.rh / row['host_rh_kpc'] - 1.
    if abs(rh_mismatch) > rh_tol:
        raise ValueError('rh mismatch: rebuilt %.3f vs CSV %.3f kpc (wrong z?)'
                         % (halo.rh, row['host_rh_kpc']))

    potential, labels, stellar = [halo], [row['host_halo_kind']], []
    if row['host_stellar_model'] == 'disk+spheroid':
        d = MN(row['host_Md_Msun'], row['host_a_kpc'], row['host_b_kpc'])
        stellar.append(d); labels.append('MN')
        if _ok(row['host_Mb_Msun']):
            stellar.append(HernquistSafe(row['host_Mb_Msun'], row['host_ab_kpc']))
            labels.append('Hq_bulge')
        if _ok(row['host_Me_Msun']):
            stellar.append(HernquistSafe(row['host_Me_Msun'], row['host_ae_kpc']))
            labels.append('Hq_env')
    elif row['host_stellar_model'] == 'spheroid':
        stellar.append(HernquistSafe(row['host_Msph_Msun'], row['host_asph_kpc']))
        labels.append('Hq_sph')
    else:
        raise ValueError('unknown host_stellar_model %r' % row['host_stellar_model'])
    potential += stellar

    # gas part of the host, used only through .rho() in ev.lram.  One HDF5
    # load serves both measured terms.
    gpotential, hot_note, cold_note = [], 'none', 'none'
    host_data, host_err = None, None
    if hot_gas in ('tabulated', 'scaled') or cold_gas in ('tabulated', 'scaled'):
        try:
            host_data = load_row_host(row, profiles_dir or PROFILES_DIR_DEFAULT)
        except Exception as exc:                 # missing file / host group
            host_err = repr(exc)[:80]

    # --- corona
    if hot_gas == 'tabulated':
        try:
            if host_data is None:
                raise RuntimeError(host_err)
            tab = tabulated_gas(host_data)
            gpotential.append(tab)
            labels.append('TabGas')
            hot_note = 'tabulated(%d bins, %.3g-%.3g kpc, slope %.2f)' % (
                tab.n_bins, tab.r_min, tab.r_max, tab.outer_slope)
        except Exception as exc:                 # too few hot-gas bins, ...
            # No corona.  Every such host in the catalogue has < 3 bins with
            # hot gas and a hot fraction below 0.5% (median 0), i.e. it has no
            # corona in the data; a beta fit never exists for them either.
            hot_gas, hot_note = 'none', 'none[fallback: %s]' % repr(exc)[:80]
    if hot_gas == 'scaled':
        try:
            if host_data is None:
                raise RuntimeError(host_err)
            m_ = host_data['fit_mask']
            f_hot = float(host_data['M_gas_hot'][m_][-1] / host_data['M_halo'][m_][-1])
            if f_hot > 0.:
                gpotential.append(scaled_halo_gas(row, z_host, f_hot))
                labels.append('ScaledHaloGas')
                hot_note = 'scaled(f_hot=%.3g)' % f_hot
            else:
                hot_note = 'none[scaled: M_hot = 0 inside the aperture]'
        except Exception as exc:
            hot_gas, hot_note = 'none', 'none[fallback: %s]' % repr(exc)[:80]
    if hot_gas == 'fgh':
        if fgh > 0.:
            gpotential.append(scaled_halo_gas(row, z_host, fgh))
            labels.append('coronagas(fgh=%g)' % fgh)
            hot_note = 'fgh(%g)' % fgh
        else:
            hot_note = 'none[--hot-gas fgh needs --fgh > 0]'
    if hot_gas == 'beta':
        if _ok(row['host_hot_rho0']):
            gpotential.append(BetaGas(row['host_hot_rho0'], row['host_hot_rc_kpc'],
                                      row['host_hot_beta']))
            labels.append('BetaGas')
            hot_note = 'beta'
        else:
            hot_note = 'none[no beta fit in the CSV]'

    # --- ISM
    if cold_gas == 'tabulated':
        try:
            if host_data is None:
                raise RuntimeError(host_err)
            tab2 = tabulated_cold_gas(host_data)
            gpotential.append(tab2)
            labels.append('TabGas2D')
            cold_note = 'tabulated(%d cells + %d filled, M=%.3g, R<=%.3g kpc, slope %.2f)' % (
                tab2.n_cells, tab2.n_filled, tab2.M_grid, tab2.R_max, tab2.outer_slope)
        except Exception as exc:                 # no 2D grid, no cold gas, ...
            # No ISM at all.  NOT the scaled model: its M_cold/M_star is
            # measured inside the whole fit aperture (up to ~2 Mpc for
            # clusters) while the grid that failed covers R,|z| <= ~180 kpc,
            # so for a host whose cold gas is all in the outskirts the scaled
            # model would spread that mass along the STELLAR profile, i.e.
            # into the central kpc where the grid says there is none (row
            # 1305 of the catalogue: 300x the measured inner gas fraction).
            cold_gas, cold_note = 'none', 'none[fallback: %s]' % repr(exc)[:80]
    if cold_gas == 'scaled':
        try:
            if host_data is None:
                raise RuntimeError(host_err)
            sg_ = scaled_cold_gas(host_data, _baryons_from_row(row))
            gpotential.append(sg_)
            labels.append('ScaledGas')
            f_cold = getattr(sg_, 'f_cold', np.nan)
            cold_note = 'scaled(f_cold=%.3g)' % f_cold
        except Exception as exc:
            cold_gas, cold_note = 'none', 'none[fallback: %s]' % repr(exc)[:80]
    if cold_gas == 'fgd':
        if fgd > 0.:  # test_evolve's gMN = MN(Md*fgd, aMN, bMN), generalised
            for comp in stellar:
                if isinstance(comp, MN):
                    gpotential.append(MN(comp.Md * fgd, comp.a, comp.b))
                else:
                    gpotential.append(HernquistSafe(comp.Mb * fgd, comp.a))
            labels.append('coldgas(fgd=%g)' % fgd)
            cold_note = 'fgd(%g)' % fgd
        else:
            cold_note = 'none[--cold-gas fgd needs --fgd > 0]'
    return potential, gpotential, dict(labels=labels, z_host=z_host, rh_mismatch=rh_mismatch,
                                       hot_gas=hot_note, cold_gas=cold_note)


# ======================================================================
# satellite
# ======================================================================

def dekel_c_from_rhm(mv, rhm, alpha, z, c_lo=1., c_hi=100.):
    """Dekel concentration for which M(<rhm) = mv/2; None if no root in [c_lo, c_hi]."""
    if not (_ok(rhm) and _ok(mv)):
        return None
    f = lambda lgc: Dekel(mv, 10.**lgc, alpha, z=z).M(rhm) - 0.5 * mv
    a, b = np.log10(c_lo), np.log10(c_hi)
    if f(a) * f(b) > 0.:
        return None
    return 10.**brentq(f, a, b, rtol=1e-4)


def dekel_c_from_DM14(mv, alpha, z):
    """Dutton & Maccio 14 c_-2(M200c, z), converted to the Dekel c = rh/rs."""
    c2 = 10.**gh.lgc2_DM14(mv, z)
    return c2 * (2. - alpha)**2 / 2.25          # rs = 2.25 r_-2 / (2-alpha)^2


def build_satellite(row, snap_z=None, alpha0=cfg.eps, c_from='rhm', msat='current',
                    fg_min=1e-4):
    """
    Satellite profiles from one CSV row.

    The structure (mv0, Rhm) is measured at dw_max_snapnum, so c0 is solved
    at that redshift; the resulting physical profile (lmax0, vmax0) is then
    re-expressed at the row's z through ev.Dekel, exactly as the evolution
    loop does, so that Delta/rh bookkeeping is consistent with the host.

    Returns a dict with the Dekel DM profile `s`, the gas profile `sg` (None
    when the dwarf has no gas), the infall quantities test_evolve.py keeps
    (mv0, lmax0, vmax0, alpha0, ms0, leff0, mmax0, lefflmax0) and `notes`.
    """
    z = float(row['z'])
    z_max = _z_of(snap_z, row['dw_max_snapnum'], z)
    notes = []
    mv0 = float(row['max_Mtot_dw_Msun'])
    c0 = dekel_c_from_rhm(mv0, row['max_Rhm_tot_dw_kpc'], alpha0, z_max) if c_from == 'rhm' else None
    if c0 is None:
        c0 = dekel_c_from_DM14(mv0, alpha0, z_max)
        notes.append('c0:DM14')
    else:
        notes.append('c0:rhm')
    s = Dekel(mv0, c0, alpha0, z=z_max)          # physical profile at infall

    fg0 = float(row['max_Mgas_dw_Msun']) / mv0
    ms0 = float(row['max_M_dw_Msun'])
    leff0 = float(row['max_Rhm_dw_kpc'])
    if not _ok(leff0):
        leff0 = gh.Reff(s.rh, s.rh / s.r2)       # test_evolve's fallback
        notes.append('leff0:Reff')
    lmax0, vmax0 = s.rmax, s.Vcirc(s.rmax)
    mmax0 = s.M(lmax0)

    # starting masses
    if msat == 'peak':
        m_start, mg_start = mv0, fg0 * mv0
    elif msat == 'current':
        m_start = min(float(row['M_dw_tot_Msun']), mv0)
        mg_start = min(float(row['M_dw_gas_Msun']), m_start)
        if m_start < mv0:
            notes.append('prestripped:%.2f' % (m_start / mv0))
    else:
        raise ValueError(msat)
    # (pre-strip along the Penarrubia+10 track if m_start < mv0, and) move the
    # bookkeeping redshift z_max -> z: same physical profile, new (c, Delta)
    c, Delta = ev.Dekel(m_start, mv0, lmax0, vmax0, alpha0, z=z)
    s = Dekel(m_start, c, alpha0, Delta=Delta, z=z)

    if mg_start / m_start > fg_min:
        sg = Dekel(mg_start, s.ch, alpha0, Delta=s.Deltah * (mg_start / m_start), z=z)
    else:
        sg = None
        notes.append('no_gas')

    return dict(s=s, sg=sg, mv0=mv0, c0=c0, c0_DM14=dekel_c_from_DM14(mv0, alpha0, z_max),
                alpha0=alpha0, fg0=fg0, ms0=ms0,
                leff0=leff0, lmax0=lmax0, vmax0=vmax0, mmax0=mmax0,
                lefflmax0=leff0 / lmax0, m_start=m_start, mg_start=mg_start,
                z_max=z_max, rh0=s.rh, notes=notes)


# ======================================================================
# initial orbit
# ======================================================================

def initial_orbit(row, inc_deg, vr_sign):
    """
    xv0 = [R, phi, z, VR, Vphi, Vz] in [kpc, rad, kpc, kpc/Gyr x3].

    The satellite is placed on the line of nodes (phi=0, z=0, R=D).  The
    tangential speed follows from the two-body (a, e) in the CSV; the radial
    speed is what is left of |V|, with sign `vr_sign` (-1 approaching).
    The orbital plane is inclined by `inc_deg` to the host disk (0 =
    prograde in-plane, 90 = polar, 180 = retrograde in-plane).
    """
    D = float(row['D_kpc'])
    V = float(row['Vrel_kms']) * cfg.kms2kpcGyr
    a, e = float(row['SMA_kpc']), float(row['ECC'])
    with np.errstate(invalid='ignore', divide='ignore'):
        # L^2 = GM a (1-e^2) and V^2 = GM (2/D - 1/a)  ->  V_tan/V; |a| for e > 1
        ratio = abs(a) * np.sqrt((1. - e**2) / (D * (2. * a - D)))
    if not np.isfinite(ratio):
        ratio = 1.                                   # no (a,e): assume tangential
    ratio = min(ratio, 1.)
    Vtan = ratio * V
    Vrad = vr_sign * np.sqrt(max(V**2 - Vtan**2, 0.))
    i = np.radians(inc_deg)
    xv0 = np.array([D, 0., 0., Vrad, Vtan * np.cos(i), Vtan * np.sin(i)])
    return xv0, dict(Vtan_over_V=ratio)


# ======================================================================
# ram-pressure radius with a robust root (workaround for ev.lram)
# ======================================================================

def lram_robust(sg, sp, gpotential, xv, kappa=1., n_grid=60):
    """
    Same equation as evolve.lram / Findlrp, but scanned on a log grid in
    [cfg.Rres, 10 sg.rh] and solved at the OUTERMOST sign change.

    evolve.lram brackets brentq on the two end points and returns Rres
    (= fully stripped) whenever both have the same sign.  For a cored
    satellite gas profile (Dekel alpha ~ 0, test_evolve's choice) the
    restoring term G m(l) rho(l) / l vanishes at both ends, so the upstream
    routine strips all gas even when the ram pressure never exceeds the
    restoring force.  Here: no sign change and restoring > ram pressure
    somewhere -> no stripping (returns 10 rh); restoring < ram pressure
    everywhere -> Rres.

    The library function is left unchanged; this only replaces it inside
    evolve_pair (default) and --upstream-lram switches back to evolve.mgas.
    """
    lgrid = np.logspace(np.log10(cfg.Rres), np.log10(10. * sg.rh), n_grid)
    f = np.array([ev.Findlrp(l, sg, sp, gpotential, xv, kappa) for l in lgrid])
    if np.all(f < 0.):
        return cfg.Rres
    pos = np.where(f > 0.)[0]
    if pos[-1] == n_grid - 1:
        return lgrid[-1]
    k = pos[-1]
    return brentq(ev.Findlrp, lgrid[k], lgrid[k + 1],
                  args=(sg, sp, gpotential, xv, kappa), rtol=1e-5, maxiter=1000)


def mgas_robust(sg, sp, gpotential, potential, xv, dt, kappa=1., alpha=1.):
    """evolve.mgas with lram_robust in place of evolve.lram."""
    lrp = lram_robust(sg, sp, gpotential, xv, kappa)
    if lrp < sg.rh:
        dm = alpha * (sg.Mh - sg.M(lrp)) * dt / pr.tdyn(potential, xv[0], xv[2])
        m = max(sg.Mh - dm, cfg.Mres)
    else:
        m = sg.Mh
    return m, lrp


# ======================================================================
# evolution loop (test_evolve.py, minus the plotting)
# ======================================================================

TRACK_KEYS = ['t', 'R', 'phi', 'z', 'VR', 'Vphi', 'Vz', 'r', 'V', 'm', 'mg', 'ms',
              'lt', 'lrp', 'leff', 'c', 'dmdt', 'dmgdt']


def evolve_pair(potential, gpotential, sat, xv0, timesteps, z, kappa=1.,
                use_df=True, rres=None, upstream_lram=False):
    rres = cfg.Rres if rres is None else rres
    s, sg = sat['s'], sat['sg']
    mv0, lmax0, vmax0, alpha0 = sat['mv0'], sat['lmax0'], sat['vmax0'], sat['alpha0']
    ms0, leff0, mmax0, lefflmax0 = sat['ms0'], sat['leff0'], sat['mmax0'], sat['lefflmax0']
    mgas_fn = ev.mgas if upstream_lram else mgas_robust
    do_gas = sg is not None and len(gpotential) > 0

    o = orbit(xv0)
    m = s.Mh
    mg = sg.Mh if sg is not None else 0.
    N = len(timesteps)
    tr = {k: np.full(N, np.nan) for k in TRACK_KEYS}
    tprev, i_last = 0., -1
    t_merge = t_disrupt = np.nan
    lt = lrp = np.nan

    for i, t in enumerate(timesteps):
        dt = t - tprev
        r = np.sqrt(o.xv[0]**2 + o.xv[2]**2)
        if r <= rres:                                 # test_evolve: `continue`
            t_merge = tprev
            break

        # ---evolve orbit (dynamical friction if m is given)
        o.integrate(t, potential, m if use_df else None)
        xv = o.xv
        if not np.all(np.isfinite(xv)):
            raise FloatingPointError('non-finite phase-space coordinates at t=%.3f Gyr' % t)
        if abs(o.t - t) > 1e-6 * max(t, 1e-3):
            # scipy's dopri5 gives up silently (orbit.integrate never checks
            # solver.successful()); o.t then lags the requested time
            raise RuntimeError('orbit integrator stalled at t=%.4f (wanted %.4f) Gyr, r=%.3f kpc'
                               % (o.t, t, np.sqrt(xv[0]**2 + xv[2]**2)))
        r = np.sqrt(xv[0]**2 + xv[2]**2)
        V = np.sqrt(xv[3]**2 + xv[4]**2 + xv[5]**2)

        # ---evolve subhalo mass and profile
        if m > cfg.Mres:
            m, lt = ev.msub(s, potential, xv, dt, choice='King62', alpha=1.)
            c, Delta = ev.Dekel(m, mv0, lmax0, vmax0, alpha0, z=z)
        else:
            m, lt, c, Delta = cfg.Mres, cfg.Rres, s.ch, s.Deltah
            if np.isnan(t_disrupt):
                t_disrupt = t
        dmdt = (s.Mh - m) / dt
        s = Dekel(m, c, alpha0, Delta=Delta, z=z)

        # ---evolve satellite gas and profile
        if do_gas:
            if sg.Mh > cfg.Mres:
                mg, lrp = mgas_fn(sg, s, gpotential, potential, xv, dt, kappa=kappa, alpha=1.)
            else:
                mg, lrp = cfg.Mres, cfg.Rres
            dmgdt = (sg.Mh - mg) / dt
            sg = Dekel(mg, c, alpha0, Delta=Delta * (mg / m), z=z)
        else:
            dmgdt = 0.

        # ---evolve satellite stellar size and mass
        mmax = s.M(s.rmax)
        g_leff, g_mstar = ev.g_EPW18(mmax / mmax0, alpha0, lefflmax0)
        leff = leff0 * g_leff
        ms = min(ms0 * g_mstar, m)

        # ---record
        for k, v in zip(TRACK_KEYS, [t, xv[0], xv[1], xv[2], xv[3], xv[4], xv[5], r, V,
                                     m, mg, ms, lt, lrp, leff, s.rh / s.rmax, dmdt, dmgdt]):
            tr[k][i] = np.asarray(v, dtype=float).item()
        tprev, i_last = t, i

    ok = slice(0, i_last + 1)
    r_arr = tr['r'][ok]
    summary = dict(
        t_end=tprev, n_steps_done=i_last + 1, merged=bool(np.isfinite(t_merge)),
        t_merge=t_merge, t_disrupt=t_disrupt,
        r_min=np.nanmin(r_arr) if i_last >= 0 else np.nan,
        r_max=np.nanmax(r_arr) if i_last >= 0 else np.nan,
        t_rmin=tr['t'][ok][np.nanargmin(r_arr)] if i_last >= 0 else np.nan,
        r_final=tr['r'][i_last] if i_last >= 0 else np.nan,
        m_final=tr['m'][i_last] if i_last >= 0 else np.nan,
        mg_final=tr['mg'][i_last] if i_last >= 0 else np.nan,
        ms_final=tr['ms'][i_last] if i_last >= 0 else np.nan,
        leff_final=tr['leff'][i_last] if i_last >= 0 else np.nan,
        n_peri=int(np.sum((r_arr[1:-1] < r_arr[:-2]) & (r_arr[1:-1] < r_arr[2:]))) if i_last >= 2 else 0,
    )
    return summary, tr


# ======================================================================
# driver
# ======================================================================

ID_COLS = ['sample', 'prikey', 'host_subkey', 'snapnum', 'z', 'dw_subfindid',
           'dw_max_snapnum', 'snap_offset', 'on_main_branch', 'bound', 'mass_ratio',
           'host_stellar_model', 'host_quality_ok', 'host_hot_bound_limited']


def run_row(idx, row, args, rng_state, snap_z=None):
    rng = np.random.default_rng(rng_state)
    out = {'row': idx}
    out.update({k: row[k] for k in ID_COLS})
    try:
        if not _ok(row['D_kpc']):
            raise ValueError('D_kpc = 0')
        potential, gpotential, hinfo = build_host(row, snap_z, fgd=args.fgd, fgh=args.fgh,
                                                  hot_gas=args.hot_gas, cold_gas=args.cold_gas,
                                                  profiles_dir=args.profiles_dir)
        labels = hinfo['labels']
        sat = build_satellite(row, snap_z, alpha0=args.alpha_sat, c_from=args.c_sat,
                              msat=args.msat)
        if args.inc == 'random':                       # isotropic orbital poles
            inc = np.degrees(np.arccos(rng.uniform(-1., 1.)))
        else:
            inc = float(args.inc)
        vr_sign = rng.choice([-1., 1.]) if args.vr_sign == 'random' else float(args.vr_sign)
        xv0, orb = initial_orbit(row, inc, vr_sign)

        z = float(row['z'])
        tmax = co.tlkbk(z, cfg.h, cfg.Om, cfg.OL) if args.tmax is None else args.tmax
        timesteps = np.linspace(0., tmax, args.nstep + 1)[1:]
        r0 = np.sqrt(xv0[0]**2 + xv0[2]**2)

        out.update(dict(
            host_components='+'.join(labels), z_host=hinfo['z_host'],
            host_rh_kpc=float(potential[0].rh), host_rh_mismatch=hinfo['rh_mismatch'],
            hot_gas=hinfo['hot_gas'], cold_gas=hinfo['cold_gas'],
            Vcirc_host_r0=float(Vcirc(potential, r0)), tdyn_host_r0=float(tdyn(potential, r0)),
            z_sat=sat['z_max'], rh_sat0=sat['rh0'],
            mv0=sat['mv0'], c0=sat['c0'], c0_DM14=sat['c0_DM14'], alpha0=sat['alpha0'], fg0=sat['fg0'],
            ms0=sat['ms0'], leff0=sat['leff0'], lmax0=sat['lmax0'],
            vmax0_kms=sat['vmax0'] / cfg.kms2kpcGyr,
            m_start=sat['m_start'], mg_start=sat['mg_start'], sat_notes=';'.join(sat['notes']),
            R0=xv0[0], z0=xv0[2], VR0=xv0[3], Vphi0=xv0[4], Vz0=xv0[5],
            V0_kms=row['Vrel_kms'], Vtan_over_V=orb['Vtan_over_V'], inc_deg=inc,
            vr_sign=vr_sign, tmax=tmax, nstep=args.nstep, dt=timesteps[0],
        ))
        if args.dry_run:
            return out, None
        t1 = time.time()
        summary, tr = evolve_pair(potential, gpotential, sat, xv0, timesteps, z,
                                  kappa=args.kappa, use_df=not args.no_df, rres=args.rres,
                                  upstream_lram=args.upstream_lram)
        out.update(summary)
        out['m_final_over_start'] = summary['m_final'] / sat['m_start']
        out['cpu_s'] = time.time() - t1
        out['error'] = ''
        return out, tr
    except Exception as exc:                            # keep going on bad rows
        out['error'] = repr(exc)
        return out, None


def _worker(payload):
    idx, row, args, seed, snap_z = payload
    return run_row(idx, row, args, seed, snap_z)


def select_rows(df, args):
    sel = pd.Series(True, index=df.index)
    if args.sample:
        sel &= df['sample'] == args.sample
    if args.quality_ok:
        sel &= df['host_quality_ok'].astype(bool)
    if args.bound_only:
        sel &= df['bound'] == 1
    if args.ecc_lt_1:
        sel &= df['ECC'] < 1.
    if args.main_branch_only:
        sel &= df['on_main_branch'].astype(bool)
    if args.no_offset:
        sel &= df['snap_offset'] == 0
    if args.select:
        for kv in args.select.split(','):
            k, v = kv.split('=')
            sel &= df[k].astype(str) == v
    if args.rows:
        sel &= df.index.isin([int(i) for i in args.rows.split(',')])
    out = df[sel]
    if args.limit:
        out = out.iloc[:args.limit]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv', help='TNG100_minor_mergers_satgen.csv')
    p.add_argument('-o', '--out', default='minor_mergers_summary.csv',
                   help='summary CSV, one line per input row')
    p.add_argument('--save-tracks', metavar='DIR', help='write one .npz per row with the time series')
    p.add_argument('--dry-run', action='store_true', help='build and print the inputs, do not evolve')
    p.add_argument('-v', '--verbose', action='store_true')
    # selection
    g = p.add_argument_group('row selection')
    g.add_argument('--limit', type=int)
    g.add_argument('--rows', help='comma-separated CSV row indices')
    g.add_argument('--select', help='e.g. prikey=0,host_subkey=10000000000000001,dw_subfindid=822')
    g.add_argument('--sample', choices=['central', 'satellite'])
    g.add_argument('--quality-ok', action='store_true', help='host_quality_ok only')
    g.add_argument('--bound-only', action='store_true', help='bound == 1 only')
    g.add_argument('--ecc-lt-1', action='store_true', help='ECC < 1 only')
    g.add_argument('--main-branch-only', action='store_true')
    g.add_argument('--no-offset', action='store_true', help='snap_offset == 0 only')
    # physics knobs
    g = p.add_argument_group('model choices')
    g.add_argument('--msat', choices=['current', 'peak'], default='current',
                   help='start from M_dw_tot at snapnum (pre-stripped along the P10 track) or from the peak mass')
    g.add_argument('--c-sat', choices=['rhm', 'DM14'], default='rhm',
                   help='satellite concentration from max_Rhm_tot_dw_kpc or from Dutton&Maccio14')
    g.add_argument('--alpha-sat', type=float, default=cfg.eps, help='satellite Dekel inner slope (test_evolve: cfg.eps)')
    g.add_argument('--cold-gas', choices=COLD_GAS_CHOICES, default='tabulated',
                   help='host ISM for ram pressure: measured (R,|z|) grid (default; no ISM where the '
                        'grid is empty), fitted stellar shape x measured M_cold/M_star, stellar shape '
                        'x --fgd, or none')
    g.add_argument('--fgd', type=float, default=0.,
                   help='with --cold-gas fgd: cold-gas fraction of the stellar model (test_evolve: 1e-6)')
    g.add_argument('--hot-gas', choices=HOT_GAS_CHOICES, default='tabulated',
                   help='host corona for ram pressure: measured profile from the HDF5 extraction '
                        '(default; no corona where it cannot be built), fitted halo shape x '
                        'measured M_hot/M_halo, halo shape x --fgh, the beta fit in the CSV, or none')
    g.add_argument('--fgh', type=float, default=0.,
                   help='with --hot-gas fgh: hot-gas fraction of the fitted halo (test_evolve: 0.1)')
    g.add_argument('--profiles-dir', default=PROFILES_DIR_DEFAULT,
                   help='directory holding the TNG100_Profiles_*.h5 files (default: %(default)s)')
    g.add_argument('--kappa', type=float, default=1., help='ram-pressure fudge factor')
    g.add_argument('--no-df', action='store_true', help='switch off dynamical friction')
    g.add_argument('--upstream-lram', action='store_true', help='use evolve.mgas as is (see lram_robust)')
    g.add_argument('--inc', default='random', help='orbital inclination to the host disk [deg], or "random" (isotropic)')
    g.add_argument('--vr-sign', default='random', help='-1 approaching, 1 receding, or "random"')
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--tmax', type=float, help='[Gyr]; default: lookback time from z to z=0')
    g.add_argument('--nstep', type=int, default=200)
    g.add_argument('--rres', type=float, help='merger radius [kpc]; default cfg.Rres (%g)' % cfg.Rres)
    g.add_argument('--snap-z', default=SNAP_Z_DEFAULT,
                   help='two-column "SnapNum Redshift" table (default: %(default)s)')
    g.add_argument('--nproc', type=int, default=1)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    rows = select_rows(df, args)
    print('>>> %d of %d rows selected' % (len(rows), len(df)))
    snap_z = load_snap_z(args.snap_z)
    if snap_z is None:
        print('    WARNING: no snapshot-redshift table (%s); using the row z for '
              'the host and satellite construction' % args.snap_z)
    # one seed per CSV row, derived from (--seed, row index): the drawn
    # inclination and approach sign of a row do not depend on which other
    # rows are selected in the same run
    payload = [(idx, row, args, np.random.SeedSequence([args.seed, int(idx)]), snap_z)
               for idx, row in rows.iterrows()]

    t0 = time.time()
    if args.nproc > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.nproc) as pool:
            results = pool.map(_worker, payload, chunksize=1)
    else:
        results = []
        for k, pl in enumerate(payload):
            res = _worker(pl)
            results.append(res)
            if args.verbose:
                o = res[0]
                msg = o['error'] or ('m/m0=%.3f r_min=%.2f kpc merged=%s (%.1fs)'
                                     % (o.get('m_final_over_start', np.nan), o.get('r_min', np.nan),
                                        o.get('merged', '-'), o.get('cpu_s', 0.)))
                print('  [%d/%d] row %d  host=%s  sat=%s  %s' % (
                    k + 1, len(payload), o['row'], o.get('host_components', '?'),
                    o.get('sat_notes', '?'), msg))
    print('    time = %.1f s' % (time.time() - t0))

    summary = pd.DataFrame([r[0] for r in results])
    if args.dry_run:
        pd.set_option('display.width', 200); pd.set_option('display.max_columns', 80)
        print(summary.T.to_string())
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    summary.to_csv(args.out, index=False)
    print('    -> %s' % args.out)

    if args.save_tracks:
        os.makedirs(args.save_tracks, exist_ok=True)
        for o, tr in results:
            if tr is None:
                continue
            fn = os.path.join(args.save_tracks, 'track_%d_%d_%d.npz'
                              % (o['prikey'], o['host_subkey'], o['dw_subfindid']))
            np.savez_compressed(fn, **tr, **{k: o[k] for k in ['R0', 'z0', 'VR0', 'Vphi0', 'Vz0',
                                                              'mv0', 'c0', 'alpha0', 'ms0', 'leff0']})
        print('    tracks -> %s' % args.save_tracks)
    return summary


if __name__ == '__main__':
    main()

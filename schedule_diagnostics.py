"""Schedule geometry diagnostics for the colored-noise / Fox forward process.

E0 deliverable of ``notes/RESEARCH_PROGRAM_DRIFT_AND_SCHEDULE.md``.  For every
forward process this dumps the schedule geometry in log-SNR coordinates so that
drift / kernel / normalization choices can be compared on one footing.

Quantities (see ``notes/NOTES_schedule_quantities.md`` for the full derivation):

    k(t)        drift coefficient            (given)
    K(t)        int_0^t k                    (log mean coeff)
    alpha(t)    signal coefficient           exp(K(t))
    D_eff(t)    effective diffusion          int_0^t C(t-s) exp(K(t)-K(s)) ds
    q(t)        marginal variance            q' = 2 k q + 2 D_eff,  normalized q(1)=V
    sigma(t)    sqrt(q(t))
    r(t)        overall input scale          sqrt(alpha^2 + q)      (c == 0)
    lambda(t)   log-SNR                      2 log alpha - log q
    -lambda'(t) log-SNR speed                2 D_eff / q  >= 0

Design of this script:

  * ``compute_path`` is an INDEPENDENT numpy implementation for an arbitrary
    ``k(t)`` and stationary kernel ``C(tau)`` (so it covers C == 1 and the
    centred-linear family k_a, which ``sde_lib.FoxVPSDE`` does not).
  * ``path_from_foxvpsde`` reads the schedule straight off the real
    ``FoxVPSDE`` cache.
  * ``cross_check`` runs BOTH on the cases they share (constant drift x 4
    kernels, vp_linear) and asserts agreement -- this is the actual audit of
    ``sde_lib``.  New ground (C == 1, k_a) then rides on the validated
    independent implementation.

Run from the repo root:

    python schedule_diagnostics.py --outdir schedule_diagnostics_out
"""

import argparse
import csv
import math
import os

import numpy as np
import torch

import sde_lib


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

# VE sigma_max: codebase defaults (configs/default_*_configs.py).
VE_SIGMA_MIN = 0.01
VE_SIGMA_MAX_CIFAR = 50.0
VE_SIGMA_MAX_CELEBA = 90.0

# EDM training-noise reference (configs: edm_sigma_data=0.5, p_mean=-1.2, p_std=1.2).
EDM_SIGMA_DATA = 0.5
EDM_P_MEAN = -1.2
EDM_P_STD = 1.2

# vp_linear beta schedule (matches VPSDE defaults and FoxVPSDE.vp_linear).
BETA_MIN, BETA_MAX = 0.1, 20.0
# integral_0^1 of k(t) = -beta(t)/2  ==  -(beta_min + beta_max) / 4.
VP_LINEAR_KBAR = -(BETA_MIN + BETA_MAX) / 4.0   # -5.025

N_GRID = 4001          # reporting grid
QUAD_GRID = 8192       # internal quadrature grid
REPORT_EPS = 1e-3      # reporting grid starts here (avoids the q->0, lambda->inf singularity)


# --------------------------------------------------------------------------- #
# kernel and drift function factories
# --------------------------------------------------------------------------- #

def matern12_C(ell):
    return lambda tau: np.exp(-tau / ell)


def matern32_C(ell):
    def C(tau):
        z = math.sqrt(3.0) * tau / ell
        return (1.0 + z) * np.exp(-z)
    return C


def gaussian_C(sigma_c):
    return lambda tau: np.exp(-(tau ** 2) / (2.0 * sigma_c ** 2))


def powerlaw_C(alpha, tau0):
    return lambda tau: 1.0 / ((1.0 + tau / tau0) ** alpha)


def constant_C():
    """C(tau) == 1 -- the explicit rank-one / infinite-memory kernel (E6)."""
    return lambda tau: np.ones_like(np.asarray(tau, dtype=np.float64))


def constant_k(u):
    return lambda t: np.full_like(t, float(u))


def vp_linear_k(beta_min=BETA_MIN, beta_max=BETA_MAX):
    return lambda t: -0.5 * (beta_min + t * (beta_max - beta_min))


def affine_k(k_bar, a):
    """k_a(t) = k_bar + a (t - 1/2).  int_0^1 k_a = k_bar for any a."""
    return lambda t: float(k_bar) + float(a) * (t - 0.5)


def matern32_length_from_kappa(kappa, u):
    """kappa = |u| ell / sqrt(3)  (EXPERIMENTS_REPORT convention, constant drift)."""
    return kappa * math.sqrt(3.0) / abs(u)


# --------------------------------------------------------------------------- #
# independent schedule computation
# --------------------------------------------------------------------------- #

def _cumtrapz(x, y):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


class SchedulePath:
    """All schedule quantities on a common reporting grid t in [0, 1]."""

    def __init__(self, label, group, t, k, K, alpha, D_eff, q,
                 kernel_ratio=np.nan, kbar=np.nan, terminal_variance=np.nan,
                 drift_desc="", kernel_desc=""):
        self.label = label
        self.group = group
        self.t = t
        self.k = k
        self.K = K
        self.alpha = alpha
        self.D_eff = D_eff
        self.q = np.maximum(q, 1e-300)
        self.sigma = np.sqrt(self.q)
        self.r = np.sqrt(alpha ** 2 + self.q)
        self.lam = np.log(np.maximum(alpha ** 2, 1e-300)) - np.log(self.q)
        self.neg_lam_dot = 2.0 * self.D_eff / self.q
        self.kernel_ratio = kernel_ratio          # C(1)/C(0)
        self.kbar = kbar                          # integral_0^1 k
        self.terminal_variance = terminal_variance
        self.drift_desc = drift_desc
        self.kernel_desc = kernel_desc


def compute_path(label, group, k_fn, C_fn, V=1.0, grid_size=QUAD_GRID,
                 drift_desc="", kernel_desc=""):
    """Independent numpy schedule for arbitrary k(t) and stationary C(tau)."""
    s = np.linspace(0.0, 1.0, grid_size)
    k = k_fn(s)
    K = _cumtrapz(s, k)                      # K(t) = int_0^t k
    kbar = K[-1]

    # D_eff(t_i) = int_0^{t_i} C(t_i - s') exp(K(t_i) - K(s')) ds'
    D_eff = np.zeros_like(s)
    for i in range(1, s.size):
        s_past = s[: i + 1]
        lag = s[i] - s_past
        resp = np.exp(K[i] - K[: i + 1])
        integ = C_fn(lag) * resp
        D_eff[i] = np.sum(0.5 * (integ[1:] + integ[:-1]) * np.diff(s_past))

    # q' = 2 k q + 2 D_eff, q(0)=0  ->  q(t) = 2 exp(2K) int_0^t D_eff exp(-2K)
    q = 2.0 * np.exp(2.0 * K) * _cumtrapz(s, D_eff * np.exp(-2.0 * K))

    # normalize terminal variance to V
    scale = V / q[-1]
    D_eff = D_eff * scale
    q = np.maximum(q * scale, 0.0)

    alpha = np.exp(K)
    kernel_ratio = float(C_fn(np.array([1.0]))[0] / C_fn(np.array([0.0]))[0])

    # resample to the common reporting grid (starts at REPORT_EPS, not 0)
    tr = np.linspace(REPORT_EPS, 1.0, N_GRID)
    def rs(y):
        return np.interp(tr, s, y)
    return SchedulePath(label, group, tr, rs(k), rs(K), rs(alpha), rs(D_eff),
                        rs(q), kernel_ratio=kernel_ratio, kbar=kbar,
                        terminal_variance=float(q[-1]), drift_desc=drift_desc,
                        kernel_desc=kernel_desc)


def path_from_foxvpsde(label, group, sde, drift_desc="", kernel_desc=""):
    """Read the schedule straight off the real FoxVPSDE cache."""
    tr = np.linspace(REPORT_EPS, 1.0, N_GRID)
    tt = torch.tensor(tr, dtype=torch.float64)
    alpha = sde.mean_coeff(tt).numpy()
    q = sde.marginal_variance(tt).numpy()
    k = sde.drift_coefficient(tt).numpy()
    D_eff = sde.effective_diffusion(tt).numpy()
    K = np.log(np.maximum(alpha, 1e-300))
    c0 = float(sde._kernel_correlation_np(np.array([0.0]))[0])
    c1 = float(sde._kernel_correlation_np(np.array([1.0]))[0])
    # integral_0^1 k via the drift, not via K[-1] (grid starts at 1e-3)
    kbar = float(np.trapz(k, tr)) + float(k[0]) * tr[0]
    return SchedulePath(label, group, tr, k, K, alpha, D_eff, q,
                        kernel_ratio=c1 / c0, kbar=kbar,
                        terminal_variance=float(q[-1]), drift_desc=drift_desc,
                        kernel_desc=kernel_desc)


def analytic_path(label, group, sde, eps):
    """VP / Cosine-VP / VE via marginal_prob (alpha, sigma known in closed form)."""
    tr = np.linspace(eps, 1.0, N_GRID)
    tt = torch.tensor(tr, dtype=torch.float64)
    ones = torch.ones(len(tr), 1, 1, 1, dtype=torch.float64)
    mean, std = sde.marginal_prob(ones, tt)
    alpha = mean[:, 0, 0, 0].numpy()
    sigma = std.numpy()
    if sigma.ndim > 1:
        sigma = sigma[:, 0, 0, 0]
    q = sigma ** 2
    K = np.log(np.maximum(alpha, 1e-300))
    k = np.gradient(K, tr)
    D_eff = 0.5 * (np.gradient(q, tr) - 2.0 * k * q)
    return SchedulePath(label, group, tr, k, K, alpha, D_eff, q,
                        kbar=float(np.trapz(k, tr)),
                        terminal_variance=float(q[-1]),
                        drift_desc="analytic", kernel_desc="-")


# --------------------------------------------------------------------------- #
# process registry
# --------------------------------------------------------------------------- #

def fox_constant(u, kernel, kappa=None, ell=None, gaussian_sigma=0.2,
                 power_law_alpha=1.5, power_law_tau0=0.1):
    """Build a constant-drift FoxVPSDE (for the cross-check + kernel sweep)."""
    if ell is None and kernel == "matern_3_2" and kappa is not None:
        ell = matern32_length_from_kappa(kappa, u)
    if ell is None:
        ell = 0.3
    return sde_lib.FoxVPSDE(
        u=u, drift_schedule="constant", kernel=kernel,
        matern_length_scale=ell, gaussian_sigma=gaussian_sigma,
        power_law_alpha=power_law_alpha, power_law_tau0=power_law_tau0,
        target_terminal_variance=1.0, schedule_grid_size=QUAD_GRID, N=1000)


def build_processes():
    procs = []

    # --- baselines -------------------------------------------------------- #
    procs.append(analytic_path("VP beta 0.1->20", "baseline",
                               sde_lib.VPSDE(0.1, 20.0), 1e-3))
    procs.append(analytic_path("Cosine VP", "baseline",
                               sde_lib.CosineVPSDE(), 1e-3))
    procs.append(analytic_path("VE CIFAR smax=50", "baseline",
                               sde_lib.VESDE(VE_SIGMA_MIN, VE_SIGMA_MAX_CIFAR), 1e-5))
    procs.append(analytic_path("VE CelebA smax=90", "baseline",
                               sde_lib.VESDE(VE_SIGMA_MIN, VE_SIGMA_MAX_CELEBA), 1e-5))

    # --- kernel sweep at production drift u = -5 (audited vs FoxVPSDE) --- #
    for kappa in (0.1, 1.0, 2.9, 1000.0):
        sde = fox_constant(-5.0, "matern_3_2", kappa=kappa)
        procs.append(path_from_foxvpsde(
            f"Fox M3/2 const u=-5 kappa={kappa:g}", "kernel_sweep", sde,
            drift_desc="k=-5", kernel_desc=f"matern_3_2 kappa={kappa:g}"))

    # explicit C == 1 (E6) -- independent impl only, FoxVPSDE has no such kernel
    procs.append(compute_path(
        "Fox C=1 const u=-5", "kernel_sweep",
        constant_k(-5.0), constant_C(), drift_desc="k=-5",
        kernel_desc="C(tau)=1 (infinite memory)"))

    # --- E4: normalized-FOX (kappa=1000 log-SNR path, rescaled to r == 1) --- #
    _ell1000 = matern32_length_from_kappa(1000.0, 5.0)
    procs.append(path_from_foxvpsde(
        "Fox normalized kappa=1000", "e4",
        sde_lib.FoxVPSDE(u=-5.0, drift_schedule="constant", kernel="matern_3_2",
                         matern_length_scale=_ell1000, target_terminal_variance=1.0,
                         normalize_scale=True, schedule_grid_size=QUAD_GRID, N=1000),
        drift_desc="k~(t) (rescaled)", kernel_desc="lambda_FOX kappa=1000, r==1"))

    # other kernel families at u = -5
    procs.append(path_from_foxvpsde(
        "Fox M1/2 const u=-5 ell=0.035", "kernel_family",
        fox_constant(-5.0, "matern_1_2", ell=0.034641),
        drift_desc="k=-5", kernel_desc="matern_1_2 ell=0.035 (~white)"))
    procs.append(path_from_foxvpsde(
        "Fox Gauss const u=-5 sc=0.2", "kernel_family",
        fox_constant(-5.0, "gaussian", gaussian_sigma=0.2),
        drift_desc="k=-5", kernel_desc="gaussian sigma_c=0.2"))
    procs.append(path_from_foxvpsde(
        "Fox PL const u=-5 a=1.5 t0=0.1", "kernel_family",
        fox_constant(-5.0, "power_law", power_law_alpha=1.5, power_law_tau0=0.1),
        drift_desc="k=-5", kernel_desc="power_law alpha=1.5 tau0=0.1"))

    # --- drift (u) sweep at the near-constant-kernel limit --------------- #
    for u in (-2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0):
        sde = fox_constant(u, "matern_3_2", kappa=1000.0)
        procs.append(path_from_foxvpsde(
            f"Fox M3/2 kappa=1000 u={u:g}", "u_sweep", sde,
            drift_desc=f"k={u:g}", kernel_desc="matern_3_2 kappa=1000"))

    # --- centred-linear drift family k_a(t) = kbar + a(t-1/2) ------------ #
    # kbar fixed to the vp_linear value so terminal alpha(1), lambda(1) match.
    kbar = VP_LINEAR_KBAR
    for a in (-9.95, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 9.95):
        # near-constant kernel so the kernel is not a confound
        procs.append(compute_path(
            f"Fox k_a a={a:g} (kbar={kbar:g})", "k_a_sweep",
            affine_k(kbar, a), constant_C(),
            drift_desc=f"k_a: kbar={kbar:g} a={a:g}", kernel_desc="C(tau)=1"))

    # vp_linear itself (a = -9.95), audited vs FoxVPSDE
    procs.append(path_from_foxvpsde(
        "Fox vp_linear kappa=1000", "drift_schedule",
        sde_lib.FoxVPSDE(u=-5.0, drift_schedule="vp_linear", kernel="matern_3_2",
                         matern_length_scale=matern32_length_from_kappa(1000.0, 5.0),
                         target_terminal_variance=1.0, schedule_grid_size=QUAD_GRID),
        drift_desc="k(t)=-beta(t)/2  (nominal kappa label only)",
        kernel_desc="matern_3_2 kappa=1000 (label wrt constant u=-5)"))

    return procs


# --------------------------------------------------------------------------- #
# scalar summary
# --------------------------------------------------------------------------- #

def _first_crossing(t, y, level):
    """First t where y crosses `level` (handles non-monotone y; NaN if never)."""
    d = y - level
    sc = np.where(np.diff(np.sign(d)) != 0)[0]
    if sc.size == 0:
        return float("nan")
    i = sc[0]
    if d[i + 1] == d[i]:
        return float(t[i])
    return float(t[i] - d[i] * (t[i + 1] - t[i]) / (d[i + 1] - d[i]))


def training_logsnr_bands(p, n=200001):
    t = np.linspace(p.t[0], 1.0, n)
    lam = np.interp(t, p.t, p.lam)
    return dict(mean=float(lam.mean()), std=float(lam.std()),
               frac_high=float(np.mean(lam < -5.0)),
               frac_mid=float(np.mean((lam >= -5.0) & (lam <= 5.0))),
               frac_low=float(np.mean(lam > 5.0)),
               frac_near0=float(np.mean(np.abs(lam) < 1.0)))


def scalar_summary(p):
    q, r, lam, k = p.q, p.r, p.lam, p.k
    qT = q[-1]
    q_mono = bool(np.all(np.diff(q) >= -1e-9))
    lam_mono = bool(np.all(np.diff(lam) <= 1e-9))
    bands = training_logsnr_bands(p)
    return dict(
        label=p.label, group=p.group,
        drift=p.drift_desc, kernel=p.kernel_desc,
        int_k=p.kbar,
        alpha_T=float(p.alpha[-1]), q_T=float(qT),
        logsnr_T=float(lam[-1]), snr_T=float(math.exp(lam[-1])),
        logsnr_eps=float(lam[0]), logsnr_range=float(lam[0] - lam[-1]),
        t10=_first_crossing(p.t, q, 0.10 * qT),
        t50=_first_crossing(p.t, q, 0.50 * qT),
        t90=_first_crossing(p.t, q, 0.90 * qT),
        r_min=float(r.min()), r_max=float(r.max()),
        r_excursion=float(max(r.max() - 1.0, 1.0 - r.min())),
        r_flag=bool(r.max() > 1.02 or r.min() < 0.80),
        q_monotone=q_mono, logsnr_monotone=lam_mono,
        # D_eff(0)=0 by construction; check it does not go NEGATIVE past the start
        D_eff_min=float(np.min(p.D_eff[p.t >= 0.02])),
        max_abs_k=float(np.max(np.abs(k))),
        kernel_ratio_C1_C0=p.kernel_ratio,
        train_logsnr_mean=bands["mean"], train_logsnr_std=bands["std"],
        train_frac_high=bands["frac_high"], train_frac_mid=bands["frac_mid"],
        train_frac_low=bands["frac_low"], train_frac_near0=bands["frac_near0"],
    )


# --------------------------------------------------------------------------- #
# audit: independent impl vs FoxVPSDE
# --------------------------------------------------------------------------- #

def cross_check(tol=5e-3):
    print("cross-check: independent numpy schedule vs sde_lib.FoxVPSDE")
    cases = []
    for kappa in (1.0, 1000.0):
        ell = matern32_length_from_kappa(kappa, 5.0)
        cases.append((f"const u=-5 matern32 kappa={kappa:g}", constant_k(-5.0),
                      matern32_C(ell), fox_constant(-5.0, "matern_3_2", kappa=kappa)))
    for u in (-3.0, -8.0):
        ell = matern32_length_from_kappa(1000.0, u)
        cases.append((f"const u={u:g} matern32 kappa=1000", constant_k(u),
                      matern32_C(ell), fox_constant(u, "matern_3_2", kappa=1000.0)))
    cases.append(("gaussian sc=0.2 u=-5", constant_k(-5.0), gaussian_C(0.2),
                  fox_constant(-5.0, "gaussian", gaussian_sigma=0.2)))
    cases.append(("vp_linear matern32 kappa=1000", vp_linear_k(),
                  matern32_C(matern32_length_from_kappa(1000.0, 5.0)),
                  sde_lib.FoxVPSDE(u=-5.0, drift_schedule="vp_linear",
                                   kernel="matern_3_2",
                                   matern_length_scale=matern32_length_from_kappa(1000.0, 5.0),
                                   target_terminal_variance=1.0,
                                   schedule_grid_size=QUAD_GRID)))
    ell1000 = matern32_length_from_kappa(1000.0, 5.0)
    for a in (2.5, 5.0):
        cases.append((f"affine a={a:g} matern32 kappa=1000", affine_k(VP_LINEAR_KBAR, a),
                      matern32_C(ell1000),
                      sde_lib.FoxVPSDE(u=-5.0, drift_schedule="affine",
                                       drift_k_bar=VP_LINEAR_KBAR, drift_a=a,
                                       kernel="matern_3_2", matern_length_scale=ell1000,
                                       target_terminal_variance=1.0,
                                       schedule_grid_size=QUAD_GRID)))
    ok = True
    g = np.linspace(0.02, 0.98, 2000)      # common interior grid, away from t->0
    for name, k_fn, C_fn, sde in cases:
        ind = compute_path(name, "check", k_fn, C_fn)
        ref = path_from_foxvpsde(name, "check", sde)
        iq, rq = np.interp(g, ind.t, ind.q), np.interp(g, ref.t, ref.q)
        il, rl = np.interp(g, ind.t, ind.lam), np.interp(g, ref.t, ref.lam)
        rel_q = np.max(np.abs(iq - rq)) / max(np.ptp(rq), 1e-6)
        rel_lam = np.max(np.abs(il - rl)) / max(np.ptp(rl), 1e-6)
        good = rel_q < tol and rel_lam < tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {name:38s} rel q={rel_q:.2e}  rel lambda={rel_lam:.2e}")
    return ok


def ode_identity_check(tol=2e-3):
    """dalpha/dt = k alpha ; dq/dt = 2kq + 2 D_eff for FoxVPSDE (both schedules)."""
    print("ODE identities on sde_lib.FoxVPSDE")
    ok = True
    sdes = [
        ("const u=-5 kappa=1000", fox_constant(-5.0, "matern_3_2", kappa=1000.0)),
        ("const u=-3 kappa=1", fox_constant(-3.0, "matern_3_2", kappa=1.0)),
        ("vp_linear kappa=1000",
         sde_lib.FoxVPSDE(u=-5.0, drift_schedule="vp_linear", kernel="matern_3_2",
                          matern_length_scale=matern32_length_from_kappa(1000.0, 5.0),
                          target_terminal_variance=1.0, schedule_grid_size=QUAD_GRID)),
    ]
    t = np.linspace(0.02, 0.98, 20001)
    tt = torch.tensor(t, dtype=torch.float64)
    for name, sde in sdes:
        a = sde.mean_coeff(tt).numpy()
        q = sde.marginal_variance(tt).numpy()
        k = sde.drift_coefficient(tt).numpy()
        de = sde.effective_diffusion(tt).numpy()
        r1 = np.max(np.abs(np.gradient(a, t) - k * a)) / max(np.ptp(a), 1e-6)
        r2 = np.max(np.abs(np.gradient(q, t) - (2 * k * q + 2 * de))) / max(np.ptp(q), 1e-6)
        good = r1 < tol and r2 < tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {name:26s} drift={r1:.2e}  variance={r2:.2e}")
    return ok


def uniform_logsnr_check(tol=1e-9):
    """sde_lib.FoxVPSDE.sampling_time_grid('uniform_logsnr') sanity."""
    print("uniform_logsnr grid on sde_lib.FoxVPSDE")
    ok = True
    sdes = [
        ("const u=-5 kappa=1000", fox_constant(-5.0, "matern_3_2", kappa=1000.0)),
        ("const u=-5 kappa=1", fox_constant(-5.0, "matern_3_2", kappa=1.0)),
        ("vp_linear kappa=1000  (uniform_variance INVALID here)",
         sde_lib.FoxVPSDE(u=-5.0, drift_schedule="vp_linear", kernel="matern_3_2",
                          matern_length_scale=matern32_length_from_kappa(1000.0, 5.0),
                          target_terminal_variance=1.0, schedule_grid_size=QUAD_GRID)),
    ]
    for name, sde in sdes:
        for N in (21, 1001):
            g = sde.sampling_time_grid(1e-3, grid="uniform_logsnr", N=N, dtype=torch.float64)
            lam = sde.log_snr(g).numpy()
            gn = g.numpy()
            lam_eps = sde.log_snr(torch.tensor([1e-3], dtype=torch.float64))[0].item()
            lam_T = sde.log_snr(torch.tensor([1.0], dtype=torch.float64))[0].item()
            tgt = np.linspace(lam_T, lam_eps, N)
            mono = bool(np.all(np.diff(gn) < 0))
            ends = abs(gn[0] - 1.0) < 1e-12 and abs(gn[-1] - 1e-3) < 1e-9
            unif = float(np.max(np.abs(lam - tgt)))
            good = mono and ends and unif < tol
            ok &= good
            print(f"  [{'ok ' if good else 'FAIL'}] {name:52s} N={N:5d}  "
                  f"mono={mono} ends={ends} max|lam-uniform|={unif:.1e}")
    return ok


def closed_form_check(tol=1e-3):
    """Near-constant kernel: sigma(t) -> (1 - e^{ut}) / (1 - e^{u})."""
    print("closed-form sigma at the constant-kernel limit")
    ok = True
    for u in (-3.0, -5.0, -8.0):
        p = compute_path(f"u={u}", "chk", constant_k(u), constant_C())
        closed = (1.0 - np.exp(u * p.t)) / (1.0 - math.exp(u))
        err = float(np.max(np.abs(p.sigma - closed)))
        good = err < tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] u={u:g}  max abs sigma err={err:.2e}")
    return ok


def normalize_scale_check(tol=5e-3):
    """FoxVPSDE(normalize_scale=True) keeps lambda(t), forces r == 1, D_eff >= 0.

    The 'normalized-FOX' E4 arm: the kappa=1000 winner's log-SNR path with the
    overall scale rescaled to alpha^2 + q = 1 (VP-type).
    """
    print("normalize_scale ('normalized-FOX') invariants")
    ell = matern32_length_from_kappa(1000.0, 5.0)
    base = sde_lib.FoxVPSDE(u=-5.0, drift_schedule="constant", kernel="matern_3_2",
                            matern_length_scale=ell, target_terminal_variance=1.0,
                            schedule_grid_size=8192, N=1000)
    norm = sde_lib.FoxVPSDE(u=-5.0, drift_schedule="constant", kernel="matern_3_2",
                            matern_length_scale=ell, target_terminal_variance=1.0,
                            normalize_scale=True, schedule_grid_size=8192, N=1000)
    g = np.linspace(0.02, 0.98, 4000)
    tt = torch.tensor(g, dtype=torch.float64)
    lam_b, lam_n = base.log_snr(tt).numpy(), norm.log_snr(tt).numpy()
    a_n, q_n = norm.mean_coeff(tt).numpy(), norm.marginal_variance(tt).numpy()
    k_n, de_n = norm.drift_coefficient(tt).numpy(), norm.effective_diffusion(tt).numpy()
    dlam = float(np.max(np.abs(lam_n - lam_b)))
    dr = float(np.max(np.abs(a_n ** 2 + q_n - 1.0)))
    ode = float(np.max(np.abs(np.gradient(q_n, g) - (2 * k_n * q_n + 2 * de_n)))
                / max(np.ptp(q_n), 1e-6))
    de_min = float(de_n.min())
    good = dlam < 1e-2 and dr < tol and de_min >= -1e-9
    print(f"  [{'ok ' if good else 'FAIL'}] max|lam_norm-lam_FOX|={dlam:.2e}  "
          f"max|a^2+q-1|={dr:.2e}  D_eff_min={de_min:+.2e}  moment-ODE rel={ode:.2e}")
    return good


# --------------------------------------------------------------------------- #
# PCA: dimensionality of the constant-drift kernel-induced schedule set
# --------------------------------------------------------------------------- #

def reproduce_kernel_pca(outdir):
    """Explicit re-derivation of the 'kernel space is ~1D' claim.

    Spec (all fixed here, nothing implicit):
      * drift            : constant, u in {-3, -5, -8}   (analysed per-u)
      * kernel families  : matern_1_2, matern_3_2, gaussian, power_law
      * memory scale     : 16 values, geometric, spanning near-white -> near-constant
      * normalization    : q(1) = 1 for every schedule
      * representation   : sigma^2(t) sampled on 256 points in [0, 1]
                           (alpha(t)=e^{ut} is kernel-independent at fixed u, so
                            sigma^2 is the ONLY kernel-dependent function)
      * PCA variants     : (a) raw curves
                           (b) mean-centred
                           (c) mean-centred + each curve L2-normalized
    Reports the singular-value spectrum and cumulative explained variance.
    """
    T = np.linspace(0.0, 1.0, 256)
    lines = ["# Kernel-schedule PCA (explicit re-derivation)", ""]
    lines.append("Spec: constant drift; sigma^2(t) on 256 pts; q(1)=1; "
                 "4 kernel families x 16 geometric memory scales.")
    lines.append("")

    def family_scales(family):
        if family == "matern_1_2":
            return [("ell", v, matern12_C(v)) for v in np.geomspace(0.005, 200.0, 16)]
        if family == "matern_3_2":
            return [("ell", v, matern32_C(v)) for v in np.geomspace(0.01, 400.0, 16)]
        if family == "gaussian":
            return [("sigma_c", v, gaussian_C(v)) for v in np.geomspace(0.01, 50.0, 16)]
        if family == "power_law":
            return [("tau0", v, powerlaw_C(1.5, v)) for v in np.geomspace(0.005, 200.0, 16)]
        raise ValueError(family)

    for u in (-3.0, -5.0, -8.0):
        curves = []
        for fam in ("matern_1_2", "matern_3_2", "gaussian", "power_law"):
            for _pname, _pval, C_fn in family_scales(fam):
                p = compute_path("x", "x", constant_k(u), C_fn, grid_size=4096)
                curves.append(np.interp(T, p.t, p.q))
        M = np.array(curves)                      # (n_schedules, 256)

        lines.append(f"## u = {u:g}  ({M.shape[0]} schedules)")
        for variant, X in (
            ("raw", M.copy()),
            ("mean-centred", M - M.mean(0)),
            ("centred + L2-normalized", None),
        ):
            if X is None:
                Xc = M - M.mean(0)
                nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
                X = Xc / np.where(nrm > 0, nrm, 1.0)
            sv = np.linalg.svd(X, compute_uv=False)
            ev = sv ** 2
            ev = ev / ev.sum()
            cum = np.cumsum(ev)
            lines.append(f"- **{variant}**: PC1={ev[0]*100:.2f}%  "
                         f"PC1+2={cum[1]*100:.2f}%  PC1..3={cum[2]*100:.2f}%")
        lines.append("")

    lines.append("Interpretation: report whatever the numbers say. PC1 >> rest "
                 "supports 'kernel is a near-1D lever at fixed constant drift'; "
                 "otherwise the earlier 97.5% claim does not reproduce.")
    path = os.path.join(outdir, "kernel_pca.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[2:]))
    return path


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

CURVE_KEYS = ("k", "K", "alpha", "sigma", "q", "r", "lam", "neg_lam_dot", "D_eff")


def write_curves_csv(path, procs):
    common = np.linspace(0.0, 1.0, N_GRID)
    header = ["t"]
    cols = [common]
    for p in procs:
        safe = (p.label.replace(" ", "_").replace("/", "").replace("=", "")
                .replace("(", "").replace(")", "").replace(",", ""))
        for key in CURVE_KEYS:
            header.append(f"{safe}_{key}")
            cols.append(np.interp(common, p.t, getattr(p, key)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(N_GRID):
            w.writerow([f"{c[i]:.8g}" for c in cols])


def write_summary_md(path, rows):
    lines = ["# Schedule diagnostics summary", ""]
    lines.append("Terminal log-SNR `lambda(1) = 2*int_0^1 k - log q(1)` "
                 "(NOT `2u` in general; `2u - log V` only for constant `k=u`). "
                 "`train_frac_*` = fraction of `t ~ U[eps,1]` training mass with "
                 "`lambda < -5` (high noise) / `|lambda| <= 5` (mid) / `> 5` (low). "
                 "For `vp_linear` the `kappa` label is nominal (defined via a "
                 "constant `u`), not the same physical parameterization.")
    lines.append("")
    cols = ["label", "int_k", "logsnr_T", "logsnr_range", "t50",
            "r_min", "r_max", "r_flag", "q_monotone", "logsnr_monotone",
            "D_eff_min", "max_abs_k", "kernel_ratio_C1_C0",
            "train_frac_high", "train_frac_mid", "train_frac_low"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, bool):
                cells.append(str(v))
            elif isinstance(v, float):
                if v != v:
                    cells.append("nan")
                elif c == "snr_T":
                    cells.append(f"{v:.2e}")
                else:
                    cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_plots(outdir, procs, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {p.label: p for p in procs}
    row_by = {r["label"]: r for r in rows}

    # ---- figure 1: u sweep --------------------------------------------------
    u_labels = sorted([l for l in by if l.startswith("Fox M3/2 kappa=1000 u=")],
                      key=lambda l: float(l.split("u=")[1]))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Constant-drift kappa=1000: u sets the schedule")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(u_labels)))
    for col, l in zip(colors, u_labels):
        p = by[l]
        u = float(l.split("u=")[1])
        axs[0, 0].plot(p.t, p.lam, color=col, lw=1.7, label=f"u={u:g}")
        axs[0, 1].plot(p.t, p.q, color=col, lw=1.7)
        axs[0, 2].plot(p.t, p.r, color=col, lw=1.7)
        axs[1, 0].plot(p.t, p.neg_lam_dot, color=col, lw=1.7)
        tt = np.linspace(p.t[0], 1.0, 40000)
        axs[1, 1].hist(np.interp(tt, p.t, p.lam), bins=120, range=(-22, 14),
                       histtype="step", color=col, density=True, lw=1.5)
    vp = by["VP beta 0.1->20"]
    for ax, key in ((axs[0, 0], "lam"), (axs[0, 1], "q"), (axs[0, 2], "r"),
                    (axs[1, 0], "neg_lam_dot")):
        ax.plot(vp.t, getattr(vp, key), "k--", lw=2.0, label="VP")
    axs[1, 1].hist(np.interp(np.linspace(vp.t[0], 1, 40000), vp.t, vp.lam),
                   bins=120, range=(-22, 14), histtype="step", color="k",
                   ls="--", density=True, lw=1.8)
    us = [float(l.split("u=")[1]) for l in u_labels]
    axs[1, 2].plot(us, [row_by[l]["t50"] for l in u_labels], "o-", color="#1b9e77")
    axs[1, 2].axhline(row_by["VP beta 0.1->20"]["t50"], color="k", ls="--")
    axs[1, 2].set_xlabel("u"); axs[1, 2].set_ylabel("t50"); axs[1, 2].set_title("front-loading vs u")
    axs[0, 0].set_title("log-SNR lambda(t)"); axs[0, 0].set_ylim(-22, 13); axs[0, 0].legend(fontsize=8)
    axs[0, 1].set_title("variance q(t)")
    axs[0, 2].set_title("scale r(t)")
    axs[1, 0].set_title("log-SNR speed -lambda'(t)  (t->0 clipped)"); axs[1, 0].set_ylim(0, 60)
    axs[1, 1].set_title("training log-SNR density (t ~ U)"); axs[1, 1].set_xlabel("lambda")
    for ax in axs.flat:
        ax.grid(alpha=0.25); ax.set_xlabel(ax.get_xlabel() or "t")
    fig.savefig(os.path.join(outdir, "u_sweep_kappa1000.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "u_sweep_kappa1000.pdf")); plt.close(fig)

    # ---- figure 2: k_a centred-linear drift family ------------------------
    ka = sorted([l for l in by if l.startswith("Fox k_a a=")],
                key=lambda l: float(l.split("a=")[1].split(" ")[0]))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(f"Centred-linear drift k_a(t) = {VP_LINEAR_KBAR:g} + a(t-1/2), C(tau)=1  "
                 "(fixed endpoints, varying front-loading)")
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(ka)))
    for col, l in zip(colors, ka):
        p = by[l]
        a = float(l.split("a=")[1].split(" ")[0])
        axs[0, 0].plot(p.t, p.k, color=col, lw=1.7, label=f"a={a:g}")
        axs[0, 1].plot(p.t, p.lam, color=col, lw=1.7)
        axs[0, 2].plot(p.t, p.r, color=col, lw=1.7)
        axs[1, 0].plot(p.t, p.q, color=col, lw=1.7)
        axs[1, 1].plot(p.t, p.D_eff, color=col, lw=1.7)
        tt = np.linspace(p.t[0], 1.0, 40000)
        axs[1, 2].hist(np.interp(tt, p.t, p.lam), bins=120, range=(-22, 14),
                       histtype="step", color=col, density=True, lw=1.5)
    axs[0, 0].set_title("drift k_a(t)"); axs[0, 0].legend(fontsize=8, ncol=2)
    axs[0, 1].set_title("log-SNR lambda(t)"); axs[0, 1].set_ylim(-22, 13)
    axs[0, 2].set_title("scale r(t)  (overshoot = bad)")
    axs[0, 2].axhline(1.0, color="0.5", lw=0.8, ls=":")
    axs[1, 0].set_title("variance q(t)  (non-monotone for a<0)")
    axs[1, 1].set_title("D_eff(t)  (must stay >= 0)")
    axs[1, 2].set_title("training log-SNR density"); axs[1, 2].set_xlabel("lambda")
    for ax in axs.flat:
        ax.grid(alpha=0.25); ax.set_xlabel(ax.get_xlabel() or "t")
    fig.savefig(os.path.join(outdir, "k_a_sweep.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "k_a_sweep.pdf")); plt.close(fig)

    # ---- figure 3: training log-SNR distribution vs EDM ------------------
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for l in ("VP beta 0.1->20", "Cosine VP", "Fox M3/2 kappa=1000 u=-5",
              "Fox M3/2 kappa=1000 u=-3", "Fox M3/2 kappa=1000 u=-8"):
        if l not in by:
            continue
        p = by[l]
        tt = np.linspace(p.t[0], 1.0, 100000)
        ax.hist(np.interp(tt, p.t, p.lam), bins=160, range=(-24, 16),
                histtype="step", density=True, lw=1.8, label=l)
    rng = np.random.default_rng(0)
    lam_edm = 2 * math.log(EDM_SIGMA_DATA) - 2 * rng.normal(EDM_P_MEAN, EDM_P_STD, 400000)
    ax.hist(lam_edm, bins=160, range=(-24, 16), histtype="step", density=True,
            lw=2.2, ls="--", color="k", label="EDM P_mean=-1.2 P_std=1.2")
    ax.set_xlabel("log-SNR lambda"); ax.set_ylabel("training density")
    ax.set_title("Induced training log-SNR distribution (t ~ U[eps,1])")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.savefig(os.path.join(outdir, "training_logsnr_distribution.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "training_logsnr_distribution.pdf")); plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="schedule_diagnostics_out")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--skip-pca", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    checks_ok = True
    if not args.skip_checks:
        checks_ok &= ode_identity_check()
        checks_ok &= cross_check()
        checks_ok &= closed_form_check()
        checks_ok &= normalize_scale_check()
        checks_ok &= uniform_logsnr_check()
        print(f"checks: {'ALL OK' if checks_ok else 'FAILURES ABOVE'}\n")

    procs = build_processes()
    rows = [scalar_summary(p) for p in procs]

    write_curves_csv(os.path.join(args.outdir, "schedule_curves.csv"), procs)
    write_summary_md(os.path.join(args.outdir, "summary.md"), rows)

    print(f"{'process':34s} {'int k':>7s} {'lam(1)':>8s} {'range':>7s} "
          f"{'t50':>6s} {'r_min':>6s} {'r_max':>6s} {'q_mono':>7s} {'lam_mono':>9s} {'De_min':>9s}")
    for r in rows:
        print(f"{r['label']:34s} {r['int_k']:7.2f} {r['logsnr_T']:8.2f} "
              f"{r['logsnr_range']:7.2f} {r['t50']:6.3f} {r['r_min']:6.3f} "
              f"{r['r_max']:6.3f} {str(r['q_monotone']):>7s} "
              f"{str(r['logsnr_monotone']):>9s} {r['D_eff_min']:9.1e}")

    if not args.skip_pca:
        print()
        reproduce_kernel_pca(args.outdir)

    if not args.skip_plots:
        make_plots(args.outdir, procs, rows)

    print(f"\nwrote {args.outdir}/  (schedule_curves.csv, summary.md, kernel_pca.md, 3x png/pdf)")
    if not checks_ok:
        raise SystemExit("self-checks FAILED")


if __name__ == "__main__":
    main()

"""Regression tests for FoxVPSDE schedule geometry and sampling grids.

Covers the E0 deliverables of notes/RESEARCH_PROGRAM_DRIFT_AND_SCHEDULE.md:
the marginal ODE identities, backward compatibility of the constant-drift
schedule, the vp_linear drift matching VPSDE, and the uniform_logsnr grid
(including the non-monotone-variance case where uniform_variance is invalid).

Run:  python -m unittest tests.test_fox_sde
"""

import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sde_lib


ELL_KAPPA1000 = 1000.0 * math.sqrt(3.0) / 5.0     # matern_3_2, |u|=5, kappa=1000
VP_LINEAR_INT_K = -(0.1 + 20.0) / 4.0             # int_0^1 (-beta(t)/2) = -5.025


def fox_const(u=-5.0, kappa=1000.0, grid=8192, V=1.0):
    ell = kappa * math.sqrt(3.0) / abs(u)
    return sde_lib.FoxVPSDE(u=u, drift_schedule="constant", kernel="matern_3_2",
                            matern_length_scale=ell, target_terminal_variance=V,
                            schedule_grid_size=grid, N=1000)


def fox_vp_linear(grid=8192, V=1.0):
    return sde_lib.FoxVPSDE(u=-5.0, drift_schedule="vp_linear", kernel="matern_3_2",
                            matern_length_scale=ELL_KAPPA1000,
                            target_terminal_variance=V, schedule_grid_size=grid, N=1000)


def fox_affine(k_bar=-5.025, a=0.0, kappa=1000.0, grid=8192, V=1.0):
    ell = kappa * math.sqrt(3.0) / 5.0
    return sde_lib.FoxVPSDE(u=-5.0, drift_schedule="affine", kernel="matern_3_2",
                            drift_k_bar=k_bar, drift_a=a, matern_length_scale=ell,
                            target_terminal_variance=V, schedule_grid_size=grid, N=1000)


def fox_normalized(kappa=1000.0, grid=8192, V=1.0):
    ell = kappa * math.sqrt(3.0) / 5.0
    return sde_lib.FoxVPSDE(u=-5.0, drift_schedule="constant", kernel="matern_3_2",
                            matern_length_scale=ell, target_terminal_variance=V,
                            normalize_scale=True, schedule_grid_size=grid, N=1000)


class FoxScheduleTest(unittest.TestCase):

    def test_alpha_satisfies_drift_ode(self):
        """dalpha/dt = k(t) alpha(t) for both drift schedules."""
        for name, sde in (("constant", fox_const()), ("vp_linear", fox_vp_linear())):
            t = torch.linspace(0.02, 0.98, 2001, dtype=torch.float64)
            alpha = sde.mean_coeff(t).numpy()
            k = sde.drift_coefficient(t).numpy()
            dt = float(t[1] - t[0])
            deriv = (alpha[2:] - alpha[:-2]) / (2.0 * dt)
            expected = k[1:-1] * alpha[1:-1]
            rel = np.max(np.abs(deriv - expected)) / max(np.ptp(alpha), 1e-6)
            self.assertLess(rel, 2e-3, msg=name)

    def test_variance_satisfies_ode(self):
        """dq/dt = 2 k(t) q(t) + 2 D_eff(t) for both drift schedules."""
        for name, sde in (("constant", fox_const()), ("vp_linear", fox_vp_linear())):
            t = torch.linspace(0.02, 0.98, 2001, dtype=torch.float64)
            q = sde.marginal_variance(t).numpy()
            k = sde.drift_coefficient(t).numpy()
            d_eff = sde.effective_diffusion(t).numpy()
            dt = float(t[1] - t[0])
            deriv = (q[2:] - q[:-2]) / (2.0 * dt)
            expected = 2.0 * k[1:-1] * q[1:-1] + 2.0 * d_eff[1:-1]
            rel = np.max(np.abs(deriv - expected)) / max(np.ptp(q), 1e-6)
            self.assertLess(rel, 2e-3, msg=name)

    def test_vp_linear_drift_and_mean_match_vpsde(self):
        """vp_linear reproduces VPSDE's k(t) and alpha(t) (EXPERIMENTS_REPORT sec 13)."""
        fox = fox_vp_linear()
        vp = sde_lib.VPSDE(beta_min=0.1, beta_max=20.0)
        t = torch.linspace(1e-3, 1.0, 500, dtype=torch.float64)
        beta_t = 0.1 + t * (20.0 - 0.1)
        self.assertLess(float(torch.max(torch.abs(
            fox.drift_coefficient(t) - (-0.5 * beta_t)))), 1e-6)
        vp_mean = vp.marginal_prob(torch.ones(len(t), 1, 1, 1, dtype=torch.float64), t)[0][:, 0, 0, 0]
        self.assertLess(float(torch.max(torch.abs(fox.mean_coeff(t) - vp_mean))), 1e-6)

    def test_terminal_logsnr_formula(self):
        """lambda(1) = 2 int_0^1 k - log V  (= 2u - log V for constant drift)."""
        for u, V in ((-3.0, 1.0), (-5.0, 1.0), (-8.0, 1.0), (-5.0, 0.5)):
            sde = fox_const(u=u, V=V)
            lam1 = float(sde.log_snr(torch.tensor([1.0], dtype=torch.float64))[0])
            self.assertAlmostEqual(lam1, 2.0 * u - math.log(V), places=4)
        fox = fox_vp_linear(V=1.0)
        lam1 = float(fox.log_snr(torch.tensor([1.0], dtype=torch.float64))[0])
        self.assertAlmostEqual(lam1, 2.0 * VP_LINEAR_INT_K, places=3)

    def test_constant_drift_schedule_backward_compatible(self):
        """Pinned reference so a refactor cannot silently change the schedule."""
        sde = fox_const(grid=4096)
        t = torch.tensor([0.1, 0.25, 0.5, 0.9], dtype=torch.float64)
        np.testing.assert_allclose(
            sde.mean_coeff(t).numpy(),
            [0.6065306597, 0.2865047969, 0.0820849986, 0.0111089965], atol=1e-9)
        np.testing.assert_allclose(
            sde.marginal_variance(t).numpy(),
            [0.1569260666, 0.5160060974, 0.8540385035, 0.9912180215], atol=1e-9)
        np.testing.assert_allclose(
            sde.effective_diffusion(t).numpy(),
            [1.9941288412, 3.6160415454, 4.6520535251, 5.0117629455], atol=1e-8)

    def test_log_snr_strictly_decreasing(self):
        for name, sde in (("constant", fox_const()), ("vp_linear", fox_vp_linear()),
                          ("colored kappa=1", fox_const(kappa=1.0))):
            t = torch.linspace(1e-3, 1.0, 4000, dtype=torch.float64)
            lam = sde.log_snr(t).numpy()
            self.assertTrue(np.all(np.diff(lam) < 0), msg=name)

    def test_vp_linear_variance_is_non_monotone(self):
        """The reason uniform_variance is invalid for vp_linear."""
        sde = fox_vp_linear()
        t = torch.linspace(0.01, 1.0, 1000, dtype=torch.float64)
        q = sde.marginal_variance(t).numpy()
        self.assertGreater(q.max(), 1.5)                 # overshoots the terminal V=1
        self.assertFalse(np.all(np.diff(q) >= -1e-9))    # not monotone increasing
        self.assertAlmostEqual(float(q[-1]), 1.0, places=3)

    def test_uniform_logsnr_grid_is_uniform_and_monotone(self):
        for name, sde in (("constant", fox_const()), ("vp_linear", fox_vp_linear())):
            for n in (21, 101, 1001):
                g = sde.sampling_time_grid(1e-3, grid="uniform_logsnr", N=n, dtype=torch.float64)
                gn = g.numpy()
                self.assertTrue(np.all(np.diff(gn) < 0), msg=f"{name} N={n} monotone")
                self.assertAlmostEqual(gn[0], 1.0, places=10)
                self.assertAlmostEqual(gn[-1], 1e-3, places=10)
                lam = sde.log_snr(g).numpy()
                targets = np.linspace(lam[0], lam[-1], n)
                self.assertLess(float(np.max(np.abs(lam - targets))), 1e-6,
                                msg=f"{name} N={n} uniform in log-SNR")

    def test_uniform_logsnr_runs_in_float32_pipeline(self):
        sde = fox_vp_linear()
        g = sde.sampling_time_grid(1e-3, grid="uniform_logsnr", dtype=torch.float32,
                                   N=sde.N + 1)
        self.assertEqual(g.shape, (sde.N + 1,))
        self.assertTrue(torch.isfinite(g).all())
        self.assertTrue(bool(torch.all(torch.diff(g) < 0)))

    def test_uniform_logsnr_rejects_non_monotone_log_snr(self):
        """A schedule with a non-decreasing log-SNR must raise, not return garbage."""
        sde = fox_const()
        # forge a non-monotone log-SNR: choose q so that lambda = 2K - log q wiggles.
        K = sde._log_mean_coeff_cpu
        ramp = torch.linspace(0.0, 1.0, len(K), dtype=K.dtype)
        lam_bad = -20.0 * ramp + 6.0 * torch.sin(2.0 * math.pi * ramp)
        q_bad = torch.exp(2.0 * K - lam_bad)
        sde._variance_cpu = q_bad / q_bad[-1]
        sde._schedule_cache = {}
        with self.assertRaises(ValueError):
            sde.sampling_time_grid(1e-3, grid="uniform_logsnr", N=51, dtype=torch.float64)

    def test_unsupported_grid_raises(self):
        sde = fox_const()
        with self.assertRaises(ValueError):
            sde.sampling_time_grid(1e-3, grid="uniform_nonsense", N=10)

    # --- affine drift k_a(t) = k_bar + a (t - 1/2) ------------------------- #

    def test_affine_a0_matches_constant(self):
        """a = 0 reproduces the constant-drift schedule exactly."""
        aff = fox_affine(k_bar=-5.0, a=0.0)
        con = fox_const(u=-5.0)
        t = torch.linspace(1e-3, 1.0, 600, dtype=torch.float64)
        np.testing.assert_allclose(aff.mean_coeff(t).numpy(),
                                   con.mean_coeff(t).numpy(), atol=1e-7)
        np.testing.assert_allclose(aff.marginal_variance(t).numpy(),
                                   con.marginal_variance(t).numpy(), atol=1e-6)
        np.testing.assert_allclose(aff.log_snr(t).numpy(),
                                   con.log_snr(t).numpy(), atol=1e-5)

    def test_affine_moment_odes(self):
        """dalpha/dt = k alpha and dq/dt = 2kq + 2 D_eff for a != 0."""
        for a in (-5.0, 2.5, 5.0):
            sde = fox_affine(a=a)
            t = torch.linspace(0.02, 0.98, 2001, dtype=torch.float64)
            alpha = sde.mean_coeff(t).numpy()
            q = sde.marginal_variance(t).numpy()
            k = sde.drift_coefficient(t).numpy()
            d_eff = sde.effective_diffusion(t).numpy()
            dt = float(t[1] - t[0])
            da = (alpha[2:] - alpha[:-2]) / (2.0 * dt)
            dq = (q[2:] - q[:-2]) / (2.0 * dt)
            rel_a = np.max(np.abs(da - k[1:-1] * alpha[1:-1])) / max(np.ptp(alpha), 1e-6)
            rel_q = np.max(np.abs(dq - (2.0 * k[1:-1] * q[1:-1] + 2.0 * d_eff[1:-1]))) / max(np.ptp(q), 1e-6)
            self.assertLess(rel_a, 2e-3, msg=f"a={a} drift ODE")
            self.assertLess(rel_q, 2e-3, msg=f"a={a} variance ODE")

    def test_affine_drift_is_linear_with_fixed_integral(self):
        """k_a(t) is exactly k_bar + a(t-1/2); int_0^1 k_a = k_bar for any a."""
        for a in (-7.5, 0.0, 2.5, 5.0):
            sde = fox_affine(k_bar=-5.025, a=a)
            t = torch.linspace(0.0, 1.0, 501, dtype=torch.float64)
            k = sde.drift_coefficient(t).numpy()
            np.testing.assert_allclose(k, -5.025 + a * (t.numpy() - 0.5), atol=1e-9)
            lam1 = float(sde.log_snr(torch.tensor([1.0], dtype=torch.float64))[0])
            self.assertAlmostEqual(lam1, 2.0 * -5.025, places=3)

    def test_affine_rejects_positive_drift_region(self):
        """|a| >= 2|k_bar| makes k_a(t) >= 0 somewhere on [0,1] -> reject."""
        with self.assertRaises(ValueError):
            fox_affine(k_bar=-5.025, a=11.0)     # k_a(1) = -5.025 + 5.5 > 0
        with self.assertRaises(ValueError):
            fox_affine(k_bar=-5.025, a=-11.0)    # k_a(0) = -5.025 + 5.5 > 0

    def test_affine_uniform_logsnr_grid_valid(self):
        """log-SNR stays monotone for the a > 0 screen points."""
        for a in (2.5, 5.0):
            sde = fox_affine(a=a)
            t = torch.linspace(1e-3, 1.0, 4000, dtype=torch.float64)
            self.assertTrue(np.all(np.diff(sde.log_snr(t).numpy()) < 0), msg=f"a={a}")
            g = sde.sampling_time_grid(1e-3, grid="uniform_logsnr", N=101, dtype=torch.float64)
            self.assertTrue(bool(torch.all(torch.diff(g) < 0)), msg=f"a={a}")

    # --- normalize_scale ('normalized-FOX', E4) --------------------------- #

    def test_normalize_scale_is_variance_preserving(self):
        """alpha_tilde^2 + q_tilde == 1 on (0, T]."""
        sde = fox_normalized()
        t = torch.linspace(1e-2, 1.0, 800, dtype=torch.float64)
        alpha = sde.mean_coeff(t).numpy()
        q = sde.marginal_variance(t).numpy()
        np.testing.assert_allclose(alpha ** 2 + q, np.ones_like(q), atol=3e-3)

    def test_normalize_scale_preserves_logsnr(self):
        """The whole point: lambda_tilde(t) == lambda_FOX(t)."""
        base = fox_const(u=-5.0, kappa=1000.0, grid=8192)
        norm = fox_normalized(kappa=1000.0, grid=8192)
        t = torch.linspace(1e-2, 0.999, 600, dtype=torch.float64)
        np.testing.assert_allclose(norm.log_snr(t).numpy(),
                                   base.log_snr(t).numpy(), atol=1e-2)

    def test_normalize_scale_moment_odes(self):
        """The rescaled (alpha_tilde, q_tilde, k_tilde, D_eff_tilde) are self-consistent."""
        sde = fox_normalized()
        t = torch.linspace(0.03, 0.97, 2001, dtype=torch.float64)
        alpha = sde.mean_coeff(t).numpy()
        q = sde.marginal_variance(t).numpy()
        k = sde.drift_coefficient(t).numpy()
        d_eff = sde.effective_diffusion(t).numpy()
        dt = float(t[1] - t[0])
        da = (alpha[2:] - alpha[:-2]) / (2.0 * dt)
        dq = (q[2:] - q[:-2]) / (2.0 * dt)
        rel_a = np.max(np.abs(da - k[1:-1] * alpha[1:-1])) / max(np.ptp(alpha), 1e-6)
        rel_q = np.max(np.abs(dq - (2.0 * k[1:-1] * q[1:-1] + 2.0 * d_eff[1:-1]))) / max(np.ptp(q), 1e-6)
        self.assertLess(rel_a, 5e-3, msg="normalized drift ODE")
        self.assertLess(rel_q, 5e-3, msg="normalized variance ODE")

    def test_normalize_scale_logsnr_strictly_decreasing(self):
        sde = fox_normalized()
        t = torch.linspace(1e-3, 1.0, 4000, dtype=torch.float64)
        self.assertTrue(np.all(np.diff(sde.log_snr(t).numpy()) < 0))

    def test_normalize_scale_off_is_unchanged(self):
        """normalize_scale=False must not perturb the existing constant schedule."""
        on = sde_lib.FoxVPSDE(u=-5.0, drift_schedule="constant", kernel="matern_3_2",
                              matern_length_scale=ELL_KAPPA1000,
                              target_terminal_variance=1.0, schedule_grid_size=8192,
                              normalize_scale=False, N=1000)
        ref = fox_const(u=-5.0, kappa=1000.0, grid=8192)
        t = torch.tensor([0.1, 0.25, 0.5, 0.9], dtype=torch.float64)
        np.testing.assert_allclose(on.marginal_variance(t).numpy(),
                                   ref.marginal_variance(t).numpy(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()

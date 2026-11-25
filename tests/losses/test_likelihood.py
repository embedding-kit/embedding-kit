"""Unit tests for ``embkit.losses.likelihood`` (count / bounded NLLs and count VAE losses)."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from embkit.losses import (
    VAELoss, gaussian_nll, poisson_nll, nb_nll, zinb_nll, beta_nll, binomial_nll, beta_binomial_nll,
    PoissonVAELoss, NBVAELoss,
)


class TestElementwiseNLL(unittest.TestCase):
    """Each NLL is checked against ``torch.distributions`` where an equivalent exists."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.poisson(torch.rand(6, 4) * 5)
        self.mu = torch.rand(6, 4) * 3 + 0.1
        self.theta = torch.rand(6, 4) + 0.5
        self.logits = torch.randn(6, 4)

    def test_gaussian_unit_variance(self):
        mean = torch.randn(6, 4)
        self.assertTrue(torch.allclose(gaussian_nll(self.x, mean), 0.5 * (self.x - mean) ** 2))

    def test_gaussian_with_logvar_matches_normal_up_to_constant(self):
        mean, logvar = torch.randn(6, 4), torch.randn(6, 4)
        ref = -torch.distributions.Normal(mean, torch.exp(0.5 * logvar)).log_prob(self.x)
        const = 0.5 * torch.log(torch.tensor(2 * torch.pi))
        self.assertTrue(torch.allclose(gaussian_nll(self.x, mean, logvar), ref - const, atol=1e-4))

    def test_poisson_matches_distribution_up_to_log_factorial(self):
        ref = -torch.distributions.Poisson(torch.exp(self.logits)).log_prob(self.x)
        expected = ref - torch.lgamma(self.x + 1.0)
        self.assertTrue(torch.allclose(poisson_nll(self.x, self.logits), expected, atol=1e-4))

    def test_nb_matches_distribution(self):
        dist = torch.distributions.NegativeBinomial(total_count=self.theta,
                                                    logits=torch.log(self.mu) - torch.log(self.theta))
        self.assertTrue(torch.allclose(nb_nll(self.x, self.mu, self.theta), -dist.log_prob(self.x), atol=1e-4))

    def test_zinb_reduces_to_nb_without_inflation(self):
        no_pi = torch.full_like(self.mu, -30.0)
        self.assertTrue(torch.allclose(zinb_nll(self.x, self.mu, self.theta, no_pi),
                                       nb_nll(self.x, self.mu, self.theta), atol=1e-3))

    def test_zinb_inflation_lowers_zero_nll(self):
        x = torch.zeros(1, 1)
        mu, theta = torch.ones(1, 1) * 5, torch.ones(1, 1)
        self.assertLess(float(zinb_nll(x, mu, theta, torch.full((1, 1), 2.0))),
                        float(nb_nll(x, mu, theta)))

    def test_beta_matches_distribution(self):
        x = torch.rand(6, 4).clamp(0.01, 0.99)
        a, b = torch.rand(6, 4) + 0.5, torch.rand(6, 4) + 0.5
        ref = -torch.distributions.Beta(a, b).log_prob(x)
        self.assertTrue(torch.allclose(beta_nll(x, a, b), ref, atol=1e-4))

    def test_binomial_matches_distribution(self):
        n = torch.randint(0, 20, (6, 4)).float()
        k = torch.floor(n * torch.rand(6, 4))
        ref = -torch.distributions.Binomial(total_count=n, logits=self.logits).log_prob(k)
        self.assertTrue(torch.allclose(binomial_nll(k, n, self.logits), ref, atol=1e-4))

    def test_beta_binomial_matches_scipy(self):
        from scipy.stats import betabinom

        n = torch.tensor([[10.0, 4.0], [7.0, 1.0], [12.0, 6.0]], dtype=torch.float64)
        k = torch.tensor([[2.0, 4.0], [7.0, 0.0], [5.0, 3.0]], dtype=torch.float64)
        a = torch.tensor([[0.5, 2.0], [1.5, 3.0], [4.0, 0.7]], dtype=torch.float64)
        b = torch.tensor([[1.0, 0.8], [2.5, 1.0], [2.0, 5.0]], dtype=torch.float64)
        ref = -betabinom.logpmf(k.numpy(), n.numpy(), a.numpy(), b.numpy())
        self.assertTrue(torch.allclose(beta_binomial_nll(k, n, a, b), torch.as_tensor(ref), atol=1e-8))


class TestPoissonVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.x = torch.poisson(torch.rand(5, 7) * 3)
        self.recon = torch.randn(5, 7)
        self.mu, self.logvar = torch.randn(5, 3), torch.randn(5, 3)

    def test_is_vae_loss(self):
        self.assertIsInstance(PoissonVAELoss(), VAELoss)

    def test_components(self):
        loss_fn = PoissonVAELoss(beta=0.5)
        total, recon, kl = loss_fn(self.recon, self.x, self.mu, self.logvar)
        exp_recon = poisson_nll(self.x, self.recon).sum(1)
        exp_kl = VAELoss._kl_divergence(self.mu, self.logvar)
        self.assertTrue(torch.allclose(recon, exp_recon.mean()))
        self.assertTrue(torch.allclose(kl, exp_kl.mean()))
        self.assertTrue(torch.allclose(total, (exp_recon + 0.5 * exp_kl).mean()))

    def test_named_output(self):
        out = PoissonVAELoss()(self.recon, self.x, self.mu, self.logvar)
        self.assertTrue(hasattr(out, "total") and hasattr(out, "recon") and hasattr(out, "kl"))


class TestNBVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.x = torch.poisson(torch.rand(5, 7) * 3)
        self.recon = torch.randn(5, 7)
        self.mu, self.logvar = torch.randn(5, 3), torch.randn(5, 3)

    def test_shared_dispersion(self):
        loss_fn = NBVAELoss(theta=2.0)
        self.assertEqual(tuple(loss_fn.log_theta.shape), (1,))
        _, recon, _ = loss_fn(self.recon, self.x, self.mu, self.logvar)
        mean = torch.nn.functional.softplus(self.recon) + 1e-8
        expected = nb_nll(self.x, mean, torch.full_like(mean, 2.0)).sum(1).mean()
        self.assertTrue(torch.allclose(recon, expected, atol=1e-5))

    def test_featurewise_dispersion_is_trainable(self):
        loss_fn = NBVAELoss(n_features=7)
        self.assertIsInstance(loss_fn.log_theta, nn.Parameter)
        total, _, _ = loss_fn(self.recon, self.x, self.mu, self.logvar)
        total.backward()
        self.assertIsNotNone(loss_fn.log_theta.grad)
        self.assertEqual(tuple(loss_fn.log_theta.grad.shape), (7,))

    def test_fixed_dispersion_is_buffer(self):
        loss_fn = NBVAELoss(learn_dispersion=False)
        self.assertEqual(len(list(loss_fn.parameters())), 0)
        self.assertIn("log_theta", dict(loss_fn.named_buffers()))

    def test_invalid_theta(self):
        with self.assertRaises(ValueError):
            NBVAELoss(theta=0.0)

    def test_trains_with_fit_vae(self):
        from embkit.models.vae import BaseVAE
        from embkit.factory.layers import Layer
        from embkit import optimize

        x = torch.poisson(torch.rand(32, 6) * 4)
        model = BaseVAE(features=[f"g{i}" for i in range(6)], latent_dim=2,
                        encoder_layers=[Layer(8, activation="relu")], decoder_layers=[Layer(8, activation="relu")])
        loss_fn = NBVAELoss(n_features=6)
        theta_before = loss_fn.log_theta.detach().clone()
        opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-2)
        history = optimize.fit_vae(model, X=DataLoader(TensorDataset(x), batch_size=16), epochs=2,
                                   loss=loss_fn, optimizer=opt, device=torch.device("cpu"), progress=False)
        self.assertEqual(len(history["loss"]), 2)
        self.assertFalse(torch.allclose(theta_before, loss_fn.log_theta.detach()))


if __name__ == "__main__":
    unittest.main()


class TestReviewRegressions(unittest.TestCase):
    """Cases from code review: large ZINB logits, float32 beta endpoints, reduced precision."""

    def test_zinb_finite_for_large_inflation_logits(self):
        x = torch.tensor([[0.0, 3.0]])
        mu, theta = torch.full((1, 2), 2.0), torch.ones(1, 2)
        for logit in (20.0, -20.0, 80.0, -80.0):
            out = zinb_nll(x, mu, theta, torch.full((1, 2), logit))
            self.assertTrue(torch.isfinite(out).all(), logit)
        # with pi -> 1, a positive count costs about the logit itself; a zero costs ~nothing
        out = zinb_nll(x, mu, theta, torch.full((1, 2), 20.0))
        self.assertLess(float(out[0, 0]), 1e-6)
        expected_pos = 20.0 + float(nb_nll(x[:, 1:], mu[:, 1:], theta[:, 1:]))
        self.assertAlmostEqual(float(out[0, 1]), expected_pos, places=3)

    def test_zinb_gradients_finite(self):
        logit = torch.full((1, 2), 20.0, requires_grad=True)
        zinb_nll(torch.tensor([[0.0, 3.0]]), torch.full((1, 2), 2.0), torch.ones(1, 2), logit).sum().backward()
        self.assertTrue(torch.isfinite(logit.grad).all())

    def test_beta_endpoints_finite_in_float32(self):
        x = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        for a, b in ((2.0, 2.0), (1.0, 1.0), (0.5, 2.0)):
            out = beta_nll(x, torch.full((1, 2), a), torch.full((1, 2), b))
            self.assertTrue(torch.isfinite(out).all(), (a, b))

    def test_reduced_precision_computed_in_float32(self):
        torch.manual_seed(0)
        x = torch.poisson(torch.rand(64, 500) * 30)
        mu, theta = torch.rand(64, 500) * 30 + 0.5, torch.rand(64, 500) * 5 + 0.5
        bf = [t.to(torch.bfloat16) for t in (x, mu, theta)]
        out = nb_nll(*bf)
        self.assertEqual(out.dtype, torch.bfloat16)
        exact = nb_nll(*[t.double() for t in bf]).sum(1)          # same rounded inputs, float64
        rel = ((out.double().sum(1) - exact).abs() / exact.abs()).max()
        self.assertLess(float(rel), 1e-2)                         # only the final cast to bf16 is lossy

    def test_vae_losses_accept_bfloat16(self):
        bf = torch.bfloat16
        x = torch.poisson(torch.rand(4, 6) * 3).to(bf)
        recon, mu, logvar = torch.randn(4, 6).to(bf), torch.randn(4, 2).to(bf), torch.randn(4, 2).to(bf)
        for loss_fn in (PoissonVAELoss(), NBVAELoss(n_features=6)):
            out = loss_fn(recon, x, mu, logvar)
            self.assertEqual(out.total.dtype, bf)
            self.assertTrue(torch.isfinite(out.total.float()))

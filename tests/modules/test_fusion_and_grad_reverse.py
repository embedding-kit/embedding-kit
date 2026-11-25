"""Unit tests for ``embkit.modules.fusion`` and ``embkit.modules.grad_reverse``."""

import unittest

import torch
from torch import nn

from embkit.factory.core import build
from embkit.modules.fusion import product_of_experts
from embkit.modules.grad_reverse import GradientReversal, grad_reverse


class TestProductOfExperts(unittest.TestCase):
    def test_single_expert_is_identity(self):
        mu, lv = torch.randn(4, 3), torch.randn(4, 3)
        out_mu, out_lv = product_of_experts([mu], [lv])
        self.assertTrue(torch.equal(out_mu, mu) and torch.equal(out_lv, lv))

    def test_equal_variances_average_means(self):
        a, b = torch.randn(4, 3), torch.randn(4, 3)
        lv = torch.zeros(4, 3)
        mu, logvar = product_of_experts([a, b], [lv, lv])
        self.assertTrue(torch.allclose(mu, (a + b) / 2))
        self.assertTrue(torch.allclose(logvar, torch.full((4, 3), float(torch.log(torch.tensor(0.5))))))

    def test_confident_expert_dominates(self):
        a, b = torch.zeros(2, 2), torch.ones(2, 2)
        mu, _ = product_of_experts([a, b], [torch.full((2, 2), -10.0), torch.full((2, 2), 10.0)])
        self.assertTrue(torch.allclose(mu, a, atol=1e-4))

    def test_weights(self):
        a, b = torch.zeros(2, 2), torch.ones(2, 2)
        lv = torch.zeros(2, 2)
        w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mu, _ = product_of_experts([a, b], [lv, lv], weights=w, weight_eps=0.0)
        self.assertTrue(torch.allclose(mu[0], a[0]) and torch.allclose(mu[1], b[1]))

    def test_identical_experts_keep_their_mean_at_extreme_variances(self):
        # regression: an absolute precision floor used to pull this mean to ~0.41
        for lv in (20.0, -25.0, 60.0, -60.0):
            mu, logvar = product_of_experts([torch.ones(1, 1)] * 2, [torch.full((1, 1), lv)] * 2)
            self.assertAlmostEqual(float(mu), 1.0, places=5)
            self.assertAlmostEqual(float(logvar), lv - float(torch.log(torch.tensor(2.0))), places=4)

    def test_matches_closed_form_in_float64(self):
        torch.manual_seed(0)
        mus = [torch.randn(5, 3, dtype=torch.float64) for _ in range(3)]
        lvs = [torch.randn(5, 3, dtype=torch.float64) * 3 for _ in range(3)]
        prec = [torch.exp(-lv) for lv in lvs]
        var = 1.0 / sum(prec)
        mu, logvar = product_of_experts(mus, lvs)
        self.assertTrue(torch.allclose(mu, var * sum(p * m for p, m in zip(prec, mus))))
        self.assertTrue(torch.allclose(logvar, torch.log(var)))

    def test_gradients_flow_to_all_experts(self):
        mus = [torch.randn(4, 2, requires_grad=True) for _ in range(2)]
        lvs = [torch.zeros(4, 2, requires_grad=True) for _ in range(2)]
        mu, logvar = product_of_experts(mus, lvs, weights=torch.full((4, 2), 0.5))
        (mu.sum() + logvar.sum()).backward()
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in mus + lvs))

    def test_errors(self):
        with self.assertRaises(ValueError):
            product_of_experts([], [])
        with self.assertRaises(ValueError):
            product_of_experts([torch.zeros(2, 2)], [])
        with self.assertRaises(ValueError):
            product_of_experts([torch.zeros(2, 2)] * 2, [torch.zeros(2, 2)] * 2, weights=torch.ones(2, 3))


class TestGradientReversal(unittest.TestCase):
    def test_forward_identity_backward_reversed(self):
        x = torch.randn(5, requires_grad=True)
        y = grad_reverse(x, 0.5)
        self.assertTrue(torch.equal(y, x))
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.full((5,), -0.5)))

    def test_module_in_sequential(self):
        head = nn.Sequential(GradientReversal(lambd=2.0), nn.Linear(3, 1))
        x = torch.randn(4, 3, requires_grad=True)
        head(x).sum().backward()
        expected = -2.0 * head[1].weight.detach().expand(4, 3)
        self.assertTrue(torch.allclose(x.grad, expected))
        self.assertIn("lambd=2.0", repr(head[0]))

    def test_roundtrip(self):
        rebuilt = build(GradientReversal(lambd=0.3))
        self.assertIsInstance(rebuilt, GradientReversal)
        self.assertEqual(rebuilt.lambd, 0.3)


if __name__ == "__main__":
    unittest.main()

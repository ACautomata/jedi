import unittest

import torch

from jedi.models.wavelet_loss import WaveletLoss


class TestWaveletLoss(unittest.TestCase):
    def test_forward_returns_non_negative_scalar(self):
        wl = WaveletLoss(wave="db4", J=2)
        pred = torch.randn(2, 1, 8, 64, 64)
        tgt = torch.randn(2, 1, 8, 64, 64)
        loss = wl(pred, tgt)
        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_zero_loss_for_identical_inputs(self):
        wl = WaveletLoss(wave="db4", J=2)
        x = torch.randn(2, 1, 8, 64, 64)
        loss = wl(x, x)
        self.assertLess(loss.item(), 1e-5)

    def test_gradient_flows_to_input(self):
        wl = WaveletLoss(wave="db4", J=2)
        pred = torch.randn(1, 1, 4, 32, 32, requires_grad=True)
        tgt = torch.randn(1, 1, 4, 32, 32)
        loss = wl(pred, tgt)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_raises_on_4d_input(self):
        wl = WaveletLoss()
        pred = torch.randn(2, 1, 64, 64)
        with self.assertRaises(AssertionError):
            wl(pred, pred)

    def test_single_slice_volume(self):
        wl = WaveletLoss()
        pred = torch.randn(2, 1, 1, 64, 64)
        tgt = torch.randn(2, 1, 1, 64, 64)
        loss = wl(pred, tgt)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

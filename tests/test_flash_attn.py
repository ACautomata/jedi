import pytest
import torch

from jedi.models.transformer import Attention, _flash_attn
from jedi.models.vis_decoder import CrossAttention


class TestFlashAttn3Helper:
    def test_returns_none_when_not_cuda(self):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        assert _flash_attn(q, k, v) is None

    def test_returns_none_for_float32(self):
        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        assert _flash_attn(q, q, q) is None

    def test_causal_param_forwarded(self):
        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        assert _flash_attn(q, q, q, causal=True) is None


class TestAttentionSdpaFallback:
    def test_self_attention_output_shape(self):
        attn = Attention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)
        assert torch.isfinite(out).all()

    def test_causal_self_attention(self):
        attn = Attention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 10, 64)
        out = attn(x, causal=True)
        assert out.shape == (2, 10, 64)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        attn = Attention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestCrossAttentionSdpaFallback:
    def test_cross_attention_output_shape(self):
        attn = CrossAttention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 8, 64)
        ctx = torch.randn(2, 1, 64)
        out = attn(x, ctx)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_cross_attention_gradient_flows(self):
        attn = CrossAttention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 8, 64, requires_grad=True)
        ctx = torch.randn(2, 1, 64, requires_grad=True)
        out = attn(x, ctx)
        out.sum().backward()
        assert x.grad is not None
        assert ctx.grad is not None

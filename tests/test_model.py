"""FR-04 / NFR-05: forward pass, modality masking, export."""

import numpy as np
import pytest
import torch

from multimodal_transformer.model.fusion_transformer import (
    IMAGE_TOKENS,
    ModelConfig,
    MultimodalFusionTransformer,
)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return MultimodalFusionTransformer(ModelConfig(pretrained_backbone=False)).eval()


def random_batch(batch=2, points=1024):
    return torch.randn(batch, 3, 224, 224), torch.randn(batch, points, 4), torch.ones(batch)


def test_forward_returns_logits_per_class(model):
    logits = model(*random_batch(3))
    assert logits.shape == (3, 3) and torch.isfinite(logits).all()
    probs = logits.softmax(1)
    torch.testing.assert_close(probs.sum(1), torch.ones(3))


def test_masked_image_does_not_influence_prediction(model):
    """With image_valid = 0 the output must depend on LiDAR only (NFR-05)."""
    image, points, _ = random_batch()
    invalid = torch.zeros(2)
    out_a = model(image, points, invalid)
    out_b = model(torch.randn_like(image) * 10, points, invalid)
    torch.testing.assert_close(out_a, out_b)
    assert not torch.allclose(model(image, points, torch.ones(2)), out_a)


def test_mixed_validity_in_one_batch(model):
    image, points, _ = random_batch()
    out = model(image, points, torch.tensor([1.0, 0.0]))
    single = model(image[1:], points[1:], torch.zeros(1))
    torch.testing.assert_close(out[1:], single, rtol=1e-4, atol=1e-5)


def test_frozen_backbone_gets_no_gradients_and_keeps_bn_stats(model):
    model.train()
    assert not model.image_encoder.backbone.training
    bn = model.image_encoder.backbone[1]
    running_mean = bn.running_mean.clone()
    image, points, valid = random_batch()
    model(image, points, valid).sum().backward()
    assert all(p.grad is None for p in model.image_encoder.backbone.parameters())
    assert model.point_encoder.mlp[0].weight.grad is not None
    torch.testing.assert_close(bn.running_mean, running_mean)


def test_modality_dropout_only_in_training():
    torch.manual_seed(0)
    cfg = ModelConfig(pretrained_backbone=False, modality_dropout=1.0, dropout=0.0)
    model = MultimodalFusionTransformer(cfg)
    image, points, _ = random_batch()
    model.train()
    train_a = model(image, points, torch.ones(2))
    train_b = model(torch.randn_like(image), points, torch.ones(2))
    torch.testing.assert_close(train_a, train_b)  # image always dropped
    model.eval()
    assert not torch.allclose(model(image, points), model(torch.randn_like(image), points))


def test_inference_bypasses_attention_fastpath_and_restores_flag(model, monkeypatch):
    # The fast path builds the full attention matrix (4.7 GB at batch 32 on 1074 tokens).
    def fastpath_kernel(*args, **kwargs):
        raise AssertionError("TransformerEncoderLayer fast path was used")

    monkeypatch.setattr(torch, "_transformer_encoder_layer_fwd", fastpath_kernel)
    torch.backends.mha.set_fastpath_enabled(True)
    image, points, _ = random_batch(2)
    with torch.inference_mode():
        logits = model(image, points, torch.tensor([1.0, 0.0]))
    assert logits.shape == (2, 3)
    assert torch.backends.mha.get_fastpath_enabled()


def test_token_count():
    assert IMAGE_TOKENS == 49


def test_model_runs_on_fewer_point_tokens(model):
    """No positional encoding, so the point-token count is free at inference (256 tokens
    make attention ~10x cheaper). It is not neutral: the share of attention that point
    tokens get against image and CLS tokens grows with their count, which is why the
    quality with fewer tokens has to be measured (evaluate.py --num-points)."""
    image, points, valid = random_batch(2, points=256)
    out_256 = model(image, points, valid)
    assert out_256.shape == (2, 3) and torch.isfinite(out_256).all()
    # The LiDAR-only path ignores the image whatever the point count.
    no_image = torch.zeros(2)
    torch.testing.assert_close(model(image, points, no_image), model(image * 10, points, no_image))


def test_onnx_export_matches_pytorch(model, tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from multimodal_transformer.model.export import export_onnx

    path = export_onnx(model, str(tmp_path / "model.onnx"))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    image, points, _ = random_batch(3)
    valid = torch.tensor([1.0, 0.0, 1.0])
    expected = model(image, points, valid).detach().numpy()
    feeds = {"image": image.numpy(), "points": points.numpy(), "image_valid": valid.numpy()}
    (actual,) = session.run(["logits"], feeds)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)

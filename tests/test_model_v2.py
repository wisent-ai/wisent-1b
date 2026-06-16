"""Tests for WisentRNMv2 geometric concept architecture."""
import math

import torch

from wisent_1b.config import wisent_tiny_v2_config
from wisent_1b.model_v2 import WisentRNMv2, orthonormalize


def test_orthonormalize():
    x = torch.randn(3, 4, 8)
    q = orthonormalize(x)
    # Rows should be orthonormal: Q @ Q^T = I
    identity = torch.einsum("krd,ktd->krt", q, q)
    expected = torch.eye(4).unsqueeze(0).expand(3, -1, -1)
    assert torch.allclose(identity, expected, atol=1e-5)


def test_forward_shape():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    outputs = model(input_ids)
    assert outputs["logits"].shape == (B, T, config.vocab_size)
    assert outputs.get("kl_loss") is not None
    assert outputs.get("concept_trace") is None


def test_forward_with_concept_trace():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids, return_concept_trace=True)
    trace = outputs["concept_trace"]
    assert trace is not None
    assert len(trace) == config.n_layers
    for layer_state in trace:
        assert layer_state["mean"].shape == (1, config.n_concepts, config.subspace_rank)
        assert layer_state["std"].shape == (1, config.n_concepts, config.subspace_rank)
        assert layer_state["embedding"].shape == (1, config.n_concepts, config.d_concept)


def test_geometric_magnitude_control():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    controls = {"magnitude": torch.zeros(1, config.n_named_concepts)}
    controls["magnitude"][0, 0] = 2.0
    outputs = model(input_ids, controls=controls)
    assert outputs["logits"].shape == (1, 8, config.vocab_size)


def test_geometric_direction_control():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    controls = {
        "direction": torch.zeros(1, config.n_named_concepts, config.subspace_rank),
    }
    controls["direction"][0, 0, 0] = 1.5
    outputs = model(input_ids, controls=controls)
    assert outputs["logits"].shape == (1, 8, config.vocab_size)


def test_unknown_control_mode_raises():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    controls = {"bad_mode": torch.zeros(1, 1)}
    try:
        model(input_ids, controls=controls)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_deterministic_sampling():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    model.eval()
    with torch.no_grad():
        out1 = model(input_ids, deterministic=True)["logits"]
        out2 = model(input_ids, deterministic=True)["logits"]
    assert torch.allclose(out1, out2, atol=1e-6)


def test_subspace_projection_roundtrip():
    config = wisent_tiny_v2_config()
    bank = WisentRNMv2(config).concept_bank
    basis = bank.get_basis()
    # Build vectors that lie exactly in the concept subspaces.
    coords = torch.randn(2, config.n_concepts, config.subspace_rank)
    x = bank.project_from_subspace(coords, basis)
    assert x.shape == (2, config.n_concepts, config.d_concept)
    coords_hat = bank.project_to_subspace(x - bank.centroid.unsqueeze(0), basis)
    assert torch.allclose(coords_hat, coords, atol=1e-4)


def test_router_weights_sum_to_one():
    config = wisent_tiny_v2_config()
    config.use_concept_router = True
    model = WisentRNMv2(config)
    tokens = torch.randn(2, 10, config.d_model)
    weights = model.router(tokens)
    assert weights.shape == (2, config.n_concepts, 10)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_named_concept_labels():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    assert model.named_concept_labels == config.named_concepts


def test_titan_manifold_changes_named_concepts():
    config = wisent_tiny_v2_config()
    config.use_titan_manifold = True
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    model.eval()
    with torch.no_grad():
        out_with = model(input_ids, deterministic=True)["logits"]

    config.use_titan_manifold = False
    model_no = WisentRNMv2(config)
    # Copy token-stream weights so only manifold differs.
    model_no.load_state_dict(model.state_dict(), strict=False)
    model_no.eval()
    with torch.no_grad():
        out_without = model_no(input_ids, deterministic=True)["logits"]

    assert not torch.allclose(out_with, out_without, atol=1e-6)


def test_geometry_loss_returned():
    config = wisent_tiny_v2_config()
    config.use_geometry_regularization = True
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)
    assert "geometry_loss" in outputs
    assert outputs["geometry_loss"].dim() == 0
    assert outputs["geometry_loss"].item() >= 0

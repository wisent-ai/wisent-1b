"""Configuration dataclass for Rej RNM models."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class RejConfig:
    """Configuration for a Rej Representation-Native Model.

    The model maintains two streams:
      - a token stream with hidden dimension ``d_model``
      - a concept stream with ``n_concepts`` slots of dimension ``d_concept``

    The first ``n_named_concepts`` concept slots are exposed as the named
    control plane (e.g. truthfulness, uncertainty, refusal). The remaining
    ``n_concepts - n_named_concepts`` slots are latent concept dimensions.
    """

    # Vocabulary and sequence
    vocab_size: int = 32000
    max_position_embeddings: int = 4096

    # Token stream (standard transformer)
    d_model: int = 2048
    n_layers: int = 22
    n_heads: int = 16
    d_head: Optional[int] = None
    intermediate_size: Optional[int] = None
    dropout: float = 0.0

    # Concept stream
    n_concepts: int = 64
    d_concept: int = 256
    n_named_concepts: int = 8
    named_concepts: List[str] = field(
        default_factory=lambda: [
            "truthfulness",
            "uncertainty",
            "refusal",
            "toxicity",
            "instruction_following",
            "code_mode",
            "medical_risk",
            "helpfulness",
        ]
    )
    concept_dropout: float = 0.0

    # Normalization and initialization
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Training
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.d_head is None:
            assert self.d_model % self.n_heads == 0, (
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
            self.d_head = self.d_model // self.n_heads
        if self.intermediate_size is None:
            # Standard GLU-style expansion factor of ~2.75
            self.intermediate_size = int(2.75 * self.d_model)
        if self.n_named_concepts > self.n_concepts:
            raise ValueError(
                f"n_named_concepts ({self.n_named_concepts}) cannot exceed "
                f"n_concepts ({self.n_concepts})"
            )
        if len(self.named_concepts) != self.n_named_concepts:
            raise ValueError(
                f"Length of named_concepts ({len(self.named_concepts)}) must match "
                f"n_named_concepts ({self.n_named_concepts})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "RejConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "RejConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def rej_1b_config() -> RejConfig:
    """Default configuration targeting ~1B parameters."""
    return RejConfig(
        vocab_size=32000,
        max_position_embeddings=4096,
        d_model=2048,
        n_layers=22,
        n_heads=16,
        n_concepts=64,
        d_concept=256,
        n_named_concepts=8,
        dropout=0.0,
    )


def rej_tiny_config() -> RejConfig:
    """Tiny configuration for fast demos and unit tests."""
    return RejConfig(
        vocab_size=256,
        max_position_embeddings=128,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_concepts=8,
        d_concept=32,
        n_named_concepts=4,
        named_concepts=["truthfulness", "refusal", "code_mode", "uncertainty"],
        dropout=0.0,
    )


@dataclass
class RejConfigV2(RejConfig):
    """Advanced configuration for RejRNMv2.

    Extends RejConfig with subspace, probabilistic, and non-linear concept
    options. Geometry is baked into the architecture rather than applied post-hoc.
    """

    # Subspace concepts: each concept is a rank-r subspace, not a single vector.
    subspace_rank: int = 4
    normalize_subspace_basis: bool = True

    # Probabilistic concept state: maintain mean + std in subspace coordinates.
    probabilistic_concepts: bool = True
    kl_weight: float = 1e-4

    # Non-linear concept dynamics via MLP cells.
    nonlinear_concepts: bool = True
    concept_mlp_hidden: Optional[int] = None

    # Input-dependent concept router.
    use_concept_router: bool = True

    # Manifold decoder: maps subspace coordinates through a small MLP.
    use_manifold_decoder: bool = True

    # Geometric control: support magnitude, direction, uncertainty, select.
    control_modes: List[str] = field(
        default_factory=lambda: ["magnitude", "direction", "uncertainty", "select"]
    )

    # Concept alignment: train concept states to be predictable from labels.
    use_concept_alignment: bool = True
    alignment_weight: float = 1.0

    # TITAN-style steering manifold: multiple directions per named concept.
    use_titan_manifold: bool = True
    n_titan_directions: int = 4

    # Geometry-aware regularization: keep concepts on their subspace manifold.
    use_geometry_regularization: bool = True
    geometry_weight: float = 1e-3

    # Language-invariant concepts: align concept states across languages.
    use_language_invariant_concepts: bool = False
    language_invariant_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.subspace_rank > self.d_concept:
            raise ValueError(
                f"subspace_rank ({self.subspace_rank}) cannot exceed d_concept ({self.d_concept})"
            )
        if self.concept_mlp_hidden is None:
            self.concept_mlp_hidden = 4 * self.d_concept


def rej_1b_v2_config() -> RejConfigV2:
    """Advanced 1B-scale configuration with geometric concepts."""
    return RejConfigV2(
        vocab_size=32000,
        max_position_embeddings=4096,
        d_model=2048,
        n_layers=22,
        n_heads=16,
        n_concepts=64,
        d_concept=256,
        n_named_concepts=8,
        subspace_rank=8,
        probabilistic_concepts=True,
        nonlinear_concepts=True,
        use_concept_router=True,
        use_manifold_decoder=True,
        dropout=0.0,
    )


def rej_tiny_v2_config() -> RejConfigV2:
    """Tiny advanced config for fast experiments."""
    return RejConfigV2(
        vocab_size=256,
        max_position_embeddings=128,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_concepts=8,
        d_concept=64,
        n_named_concepts=4,
        named_concepts=["truthfulness", "refusal", "code_mode", "uncertainty"],
        subspace_rank=4,
        probabilistic_concepts=True,
        nonlinear_concepts=True,
        use_concept_router=True,
        use_manifold_decoder=True,
        dropout=0.0,
    )

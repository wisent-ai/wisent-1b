"""Advanced Rej RNM v2: geometric concepts baked into the architecture.

Concepts are no longer single vectors. Each concept is a subspace (basis + centroid)
optionally carrying a probabilistic state (mean + std). Concept dynamics are non-linear
MLP cells. An input-dependent router decides which concepts read from which tokens.
"""
from .concepts import (
    ConceptRouter,
    ProbabilisticConceptState,
    SubspaceConceptBank,
    orthonormalize,
)
from .dynamics import (
    ConceptAlignmentHead,
    GeometricControl,
    NonlinearConceptCell,
    SteeringManifold,
)
from .network import RejRNMv2

__all__ = [
    "ConceptAlignmentHead",
    "ConceptRouter",
    "GeometricControl",
    "NonlinearConceptCell",
    "ProbabilisticConceptState",
    "RejRNMv2",
    "SteeringManifold",
    "SubspaceConceptBank",
    "orthonormalize",
]

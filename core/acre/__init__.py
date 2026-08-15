"""Core-owned ACRE semantics and orchestration primitives."""

from .predicates import PredicateGrammar, SYNTHESIZER_VERSION, predicate_complexity
from .cegis import BoundaryObservation, StatisticalCEGIS, SynthesisResult
from .factorial import CoverageResult, FactorialBlock, FactorialEngine, FactorialEstimate, simulate_coverage
from .acquisition import AcquisitionPolicy, AcquisitionQuery, AcquisitionResult, run_acquisition
from .router import BundleCertificate, ConservativeCausalRouter, RoutingDecision
from .engine import AcreEngine
from .relation import RelationIdentifier, RelationIdentification
from .factorial import HigherOrderEstimate, ThreeWayBlock, estimate_higher_order
from core.governance import EvolutionDecision
from .evidence import EvidenceAssessment, adversarial_events, assess, representative_events
from .controller import AcreController

__all__ = [
    "PredicateGrammar", "predicate_complexity", "SynthesisResult", "StatisticalCEGIS", "BoundaryObservation", "SYNTHESIZER_VERSION",
    "CoverageResult", "FactorialBlock", "FactorialEngine", "FactorialEstimate", "simulate_coverage",
    "AcquisitionPolicy", "AcquisitionQuery", "AcquisitionResult", "run_acquisition",
    "ConservativeCausalRouter", "RoutingDecision", "BundleCertificate",
    "AcreEngine", "EvolutionDecision", "RelationIdentifier", "RelationIdentification", "HigherOrderEstimate", "ThreeWayBlock", "estimate_higher_order",
    "EvidenceAssessment", "representative_events", "adversarial_events", "assess",
    "AcreController",
]

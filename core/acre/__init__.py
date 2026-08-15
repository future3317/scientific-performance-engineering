"""Core-owned ACRE semantics and orchestration primitives."""

from .predicates import PredicateGrammar, SYNTHESIZER_VERSION, predicate_complexity
from .cegis import BoundaryObservation, StatisticalCEGIS, SynthesisResult
from .factorial import CoverageResult, FactorialBlock, FactorialEngine, FactorialEstimate, simulate_coverage
from .acquisition import AcquisitionPolicy, AcquisitionQuery, AcquisitionResult, run_acquisition
from .router import BundleCertificate, ConservativeCausalRouter, RoutingDecision, validate_relation_nonoverlap
from .engine import AcreEngine
from .relation import RelationIdentifier, RelationIdentification, relational_cegis
from .policy import RelationDecisionPolicy
from .factorial import HigherOrderEstimate, ThreeWayBlock, estimate_higher_order
from core.governance import EvolutionDecision
from .evidence import EvidenceAssessment, adversarial_events, assess, representative_events
from .controller import AcreController

__all__ = [
    "PredicateGrammar", "predicate_complexity", "SynthesisResult", "StatisticalCEGIS", "BoundaryObservation", "SYNTHESIZER_VERSION",
    "CoverageResult", "FactorialBlock", "FactorialEngine", "FactorialEstimate", "simulate_coverage",
    "AcquisitionPolicy", "AcquisitionQuery", "AcquisitionResult", "run_acquisition",
    "ConservativeCausalRouter", "RoutingDecision", "BundleCertificate", "validate_relation_nonoverlap", "RelationDecisionPolicy",
    "AcreEngine", "EvolutionDecision", "RelationIdentifier", "RelationIdentification", "relational_cegis", "HigherOrderEstimate", "ThreeWayBlock", "estimate_higher_order",
    "EvidenceAssessment", "representative_events", "adversarial_events", "assess",
    "AcreController",
]

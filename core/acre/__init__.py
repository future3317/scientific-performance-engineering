"""Core-owned ACRE semantics and orchestration primitives."""

from .predicates import PredicateGrammar, SYNTHESIZER_VERSION, predicate_complexity
from .cegis import BoundaryObservation, StatisticalCEGIS, SynthesisResult, SynthesisCertificate, synthesize_boundary, synthesize_applicability
from .factorial import (
    CoverageResult, FactorialBlock, FactorialEngine, FactorialEstimate,
    HigherOrderEstimate, ThreeWayBlock, RelationEvidenceCertificate, canonical_relation_label,
    estimate_higher_order, simulate_coverage,
)
from .acquisition import AcquisitionPolicy, AcquisitionQuery, AcquisitionResult, run_acquisition
from .router import BundleCertificate, ConservativeCausalRouter, RoutingDecision, RequiredExperiment, validate_relation_nonoverlap
from .engine import AcreEngine
from .relation import RelationIdentifier, RelationIdentification, relational_cegis
from .policy import RelationDecisionPolicy
from core.governance import EvolutionDecision
from .evidence import EvidenceAssessment, adversarial_events, assess, representative_events
from .controller import AcreController
from .maintainer import AcreMaintainer, MaintenanceInput, MaintenanceTransition
from .experiments import ExperimentExecutor, ExperimentPlan, ExperimentExecution, execute_paired_plan
from .actions import RealizationValidator

__all__ = [
    "PredicateGrammar", "predicate_complexity", "SynthesisResult", "SynthesisCertificate", "synthesize_boundary", "synthesize_applicability", "StatisticalCEGIS", "BoundaryObservation", "SYNTHESIZER_VERSION",
    "CoverageResult", "FactorialBlock", "FactorialEngine", "FactorialEstimate", "simulate_coverage",
    "AcquisitionPolicy", "AcquisitionQuery", "AcquisitionResult", "run_acquisition",
    "ConservativeCausalRouter", "RoutingDecision", "RequiredExperiment", "BundleCertificate", "validate_relation_nonoverlap", "RelationDecisionPolicy",
    "AcreEngine", "EvolutionDecision", "RelationIdentifier", "RelationIdentification", "relational_cegis", "HigherOrderEstimate", "ThreeWayBlock", "RelationEvidenceCertificate", "estimate_higher_order", "canonical_relation_label",
    "EvidenceAssessment", "representative_events", "adversarial_events", "assess",
    "AcreController", "AcreMaintainer", "MaintenanceInput", "MaintenanceTransition", "ExperimentExecutor", "ExperimentPlan", "ExperimentExecution", "execute_paired_plan", "RealizationValidator",
]

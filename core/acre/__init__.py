"""Deterministic ACRE predicate, interaction, acquisition, and routing primitives."""

from .predicate_synthesis import PredicateGrammar, SynthesisResult, SYNTHESIZER_VERSION
from .factorial import CoverageResult, FactorialBlock, FactorialEngine, FactorialEstimate, simulate_coverage
from .acquisition import AcquisitionPolicy, AcquisitionQuery, AcquisitionResult, run_acquisition
from .router import ConservativeCausalRouter, InteractionEvidence, RuleCandidate, RoutingDecision

__all__ = [
    "PredicateGrammar", "SynthesisResult", "SYNTHESIZER_VERSION",
    "CoverageResult", "FactorialBlock", "FactorialEngine", "FactorialEstimate", "simulate_coverage",
    "AcquisitionPolicy", "AcquisitionQuery", "AcquisitionResult", "run_acquisition",
    "ConservativeCausalRouter", "InteractionEvidence", "RuleCandidate", "RoutingDecision",
]

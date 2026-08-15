"""Core-owned ACRE semantics and orchestration primitives."""

from .predicates import PredicateGrammar, SYNTHESIZER_VERSION, predicate_complexity
from .cegis import BoundaryObservation, StatisticalCEGIS, SynthesisResult
from .factorial import CoverageResult, FactorialBlock, FactorialEngine, FactorialEstimate, simulate_coverage
from .acquisition import AcquisitionPolicy, AcquisitionQuery, AcquisitionResult, run_acquisition
from .router import ConservativeCausalRouter, RoutingDecision
from .engine import AcreEngine
from core.governance import EvolutionDecision

__all__ = [
    "PredicateGrammar", "predicate_complexity", "SynthesisResult", "StatisticalCEGIS", "BoundaryObservation", "SYNTHESIZER_VERSION",
    "CoverageResult", "FactorialBlock", "FactorialEngine", "FactorialEstimate", "simulate_coverage",
    "AcquisitionPolicy", "AcquisitionQuery", "AcquisitionResult", "run_acquisition",
    "ConservativeCausalRouter", "RoutingDecision",
    "AcreEngine", "EvolutionDecision",
]

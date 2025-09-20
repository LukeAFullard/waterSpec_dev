import numpy as np
from waterSpec.interpreter import (
    get_scientific_interpretation,
    get_persistence_traffic_light,
    compare_to_benchmarks,
    interpret_results
)
import pytest

def test_get_scientific_interpretation():
    assert "White noise" in get_scientific_interpretation(0)
    assert "fGn-like" in get_scientific_interpretation(0.5)
    assert "Pink noise" in get_scientific_interpretation(1.0)
    assert "fBm-like" in get_scientific_interpretation(1.5)
    assert "Brownian noise" in get_scientific_interpretation(2.0)
    assert "Warning" in get_scientific_interpretation(-1.0)

def test_get_persistence_traffic_light():
    assert "Event-driven" in get_persistence_traffic_light(0.2)
    assert "Mixed" in get_persistence_traffic_light(0.7)
    assert "Persistent" in get_persistence_traffic_light(1.5)

def test_compare_to_benchmarks():
    assert "E. coli" in compare_to_benchmarks(0.3)
    assert "Nitrate-N" in compare_to_benchmarks(1.8)
    # A beta of 0.9 falls directly in the Ortho-P range (0.6-1.2)
    assert "Ortho-P" in compare_to_benchmarks(0.9)

def test_interpret_results_basic():
    fit_results = {'beta': 1.7}
    results = interpret_results(fit_results, param_name="Nitrate")
    assert "Analysis for: Nitrate" in results['summary_text']
    assert "β = 1.70" in results['summary_text']
    assert "Persistent" in results['persistence_level']
    assert "fBm-like" in results['scientific_interpretation']
    assert "Chloride" in results['benchmark_comparison']
    assert results['uncertainty_warning'] is None

def test_interpret_results_with_ci():
    fit_results = {'beta': 1.7, 'beta_ci_lower': 1.5, 'beta_ci_upper': 1.9}
    results = interpret_results(fit_results, param_name="Nitrate")
    assert "95% CI: 1.50–1.90" in results['summary_text']
    assert results['uncertainty_warning'] is None

def test_interpret_results_with_wide_ci():
    fit_results = {'beta': 1.7, 'beta_ci_lower': 1.0, 'beta_ci_upper': 2.4}
    results = interpret_results(fit_results, param_name="Nitrate")
    assert "Warning: The confidence interval width" in results['uncertainty_warning']
    assert "large" in results['summary_text']

def test_interpret_results_no_param_name():
    fit_results = {'beta': 0.5}
    results = interpret_results(fit_results)
    assert "Analysis for: Parameter" in results['summary_text']

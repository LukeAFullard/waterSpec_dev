import logging
import numpy as np
import pytest

from waterSpec.model_selector import ModelSelector


@pytest.fixture
def dummy_data():
    return np.array([1, 2, 3]), np.array([4, 5, 6])


@pytest.fixture
def model_selector():
    return ModelSelector(logger=logging.getLogger("test_model_selector"))


def test_select_best_model_standard_wins(dummy_data, model_selector, mocker):
    freq, power = dummy_data

    mock_standard = mocker.patch(
        "waterSpec.model_selector.fit_standard_model",
        return_value={"bic": 10.0, "beta": 1.5},
    )
    mock_segmented = mocker.patch(
        "waterSpec.model_selector.fit_segmented_spectrum",
        return_value={"bic": 20.0, "betas": [1.0, 2.0], "n_breakpoints": 1},
    )

    result = model_selector.select_best_model(
        frequency=freq,
        power=power,
        fit_method="ols",
        ci_method="parametric",
        bootstrap_type="parametric",
        n_bootstraps=10,
        p_threshold=0.05,
        max_breakpoints=1,
        seed=42,
    )

    assert result["chosen_model_type"] == "standard"
    assert result["model_type"] == "standard"
    assert result["n_breakpoints"] == 0
    assert result["bic"] == 10.0
    assert "all_models" in result
    assert len(result["all_models"]) == 2

    mock_standard.assert_called_once()
    mock_segmented.assert_called_once()


def test_select_best_model_segmented_wins(dummy_data, model_selector, mocker):
    freq, power = dummy_data

    mocker.patch(
        "waterSpec.model_selector.fit_standard_model",
        return_value={"bic": 30.0, "beta": 1.5},
    )
    mocker.patch(
        "waterSpec.model_selector.fit_segmented_spectrum",
        return_value={"bic": 15.0, "betas": [1.0, 2.0], "n_breakpoints": 1},
    )

    result = model_selector.select_best_model(
        frequency=freq,
        power=power,
        fit_method="ols",
        ci_method="parametric",
        bootstrap_type="parametric",
        n_bootstraps=10,
        p_threshold=0.05,
        max_breakpoints=1,
        seed=42,
    )

    assert result["chosen_model_type"] == "segmented"
    assert result["model_type"] == "segmented_1bp"
    assert result["n_breakpoints"] == 1
    assert result["bic"] == 15.0


def test_select_best_model_standard_fails_gracefully(dummy_data, model_selector, mocker):
    freq, power = dummy_data

    # Standard model fails by returning infinite BIC and a failure reason
    mocker.patch(
        "waterSpec.model_selector.fit_standard_model",
        return_value={"bic": np.inf, "failure_reason": "Not enough data"},
    )
    mocker.patch(
        "waterSpec.model_selector.fit_segmented_spectrum",
        return_value={"bic": 15.0, "betas": [1.0, 2.0], "n_breakpoints": 1},
    )

    result = model_selector.select_best_model(
        frequency=freq,
        power=power,
        fit_method="ols",
        ci_method="parametric",
        bootstrap_type="parametric",
        n_bootstraps=10,
        p_threshold=0.05,
        max_breakpoints=1,
        seed=None,
    )

    assert result["chosen_model_type"] == "segmented"
    assert "Standard model (0 breakpoints): Not enough data" in result["failed_model_reasons"]


def test_select_best_model_all_fail(dummy_data, model_selector, mocker):
    freq, power = dummy_data

    mocker.patch(
        "waterSpec.model_selector.fit_standard_model",
        return_value={"bic": np.inf, "failure_reason": "Failed standard"},
    )
    mocker.patch(
        "waterSpec.model_selector.fit_segmented_spectrum",
        return_value={"bic": np.nan, "failure_reason": "Failed segmented"},
    )

    with pytest.raises(RuntimeError, match="All models failed; no valid BIC values found.") as exc_info:
        model_selector.select_best_model(
            frequency=freq,
            power=power,
            fit_method="ols",
            ci_method="parametric",
            bootstrap_type="parametric",
            n_bootstraps=10,
            p_threshold=0.05,
            max_breakpoints=1,
            seed=42,
        )

    assert "Failed standard" in str(exc_info.value)
    assert "Failed segmented" in str(exc_info.value)


def test_select_best_model_standard_raises_exception(dummy_data, model_selector, mocker):
    freq, power = dummy_data

    # Standard model raises an exception
    mocker.patch(
        "waterSpec.model_selector.fit_standard_model",
        side_effect=ValueError("Critical failure"),
    )
    mocker.patch(
        "waterSpec.model_selector.fit_segmented_spectrum",
        return_value={"bic": 20.0, "betas": [1.0, 2.0], "n_breakpoints": 1},
    )

    result = model_selector.select_best_model(
        frequency=freq,
        power=power,
        fit_method="ols",
        ci_method="parametric",
        bootstrap_type="parametric",
        n_bootstraps=10,
        p_threshold=0.05,
        max_breakpoints=1,
        seed=123,
    )

    assert result["chosen_model_type"] == "segmented"
    assert any("Critical failure" in reason for reason in result["failed_model_reasons"])


def test_select_best_model_segmented_raises_exception(dummy_data, model_selector, mocker):
    freq, power = dummy_data

    mocker.patch(
        "waterSpec.model_selector.fit_standard_model",
        return_value={"bic": 10.0, "beta": 1.5},
    )
    # Segmented model raises an unexpected exception
    mocker.patch(
        "waterSpec.model_selector.fit_segmented_spectrum",
        side_effect=Exception("Unexpected crash"),
    )

    result = model_selector.select_best_model(
        frequency=freq,
        power=power,
        fit_method="ols",
        ci_method="parametric",
        bootstrap_type="parametric",
        n_bootstraps=10,
        p_threshold=0.05,
        max_breakpoints=1,
        seed=123,
    )

    assert result["chosen_model_type"] == "standard"
    assert any("Unexpected crash" in reason for reason in result["failed_model_reasons"])


def test_select_best_model_multiple_breakpoints(dummy_data, model_selector, mocker):
    freq, power = dummy_data

    mocker.patch(
        "waterSpec.model_selector.fit_standard_model",
        return_value={"bic": 50.0, "beta": 1.5},
    )

    # Mock segmented to return different BIC based on n_breakpoints
    def mock_fit_segmented(*args, **kwargs):
        nbp = kwargs.get("n_breakpoints")
        if nbp == 1:
            return {"bic": 40.0, "betas": [1.0, 2.0], "n_breakpoints": 1}
        elif nbp == 2:
            return {"bic": 30.0, "betas": [1.0, 2.0, 3.0], "n_breakpoints": 2}
        return {"bic": np.inf}

    mocker.patch(
        "waterSpec.model_selector.fit_segmented_spectrum",
        side_effect=mock_fit_segmented,
    )

    result = model_selector.select_best_model(
        frequency=freq,
        power=power,
        fit_method="ols",
        ci_method="parametric",
        bootstrap_type="parametric",
        n_bootstraps=10,
        p_threshold=0.05,
        max_breakpoints=2,
        seed=123,
    )

    assert result["chosen_model_type"] == "segmented"
    assert result["model_type"] == "segmented_2bp"
    assert result["n_breakpoints"] == 2
    assert result["bic"] == 30.0
    assert len(result["all_models"]) == 3  # Standard, 1bp, 2bp

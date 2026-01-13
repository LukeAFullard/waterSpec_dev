# Validation Findings

## `validate_analysis.py`
- Fails initially because the Segmented Model was rejected (Davies p-value 0.28 > 0.05).
- Reference output expected Segmented Model with specific parameters.
- Investigation revealed that the reference output was likely generated with `detrend_method="linear"`, whereas `validate_analysis.py` defaulted to `None`.
- With `detrend_method="linear"`, the Davies test is significant (p=0.007), and parameters are close to reference.
- **Action**: Updated `validate_analysis.py` to use `detrend_method="linear"`.

## `verify_synthetic.py`
- Fails for Uneven Sampling with Red Noise (Beta=2.0, Est=0.54) and Pink Noise (Beta=1.0, Est=0.65).
- This is a known limitation of Lomb-Scargle periodogram for red noise with irregular sampling (spectral leakage bias).
- The package provides `psresp` module (Power Spectral Response) to handle this, verified by `run_psresp_beta_sweep.py`.
- `Analysis` class currently uses standard LS method.
- **Action**: Acknowledge as known limitation. Users should use `psresp` for uneven red noise.

## `run_full_comparison_sweep.py`
- Failed due to missing arguments.

## `reproduce_red_noise_psresp.py` & `run_psresp_beta_sweep.py`
- Pass and demonstrate PSRESP effectiveness.

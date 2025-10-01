# Advanced Segmented Fitting: Using Two Breakpoints

The `waterSpec` package allows for segmented (or piecewise) regression to identify points where the spectral slope changes. While the default is to find a single breakpoint, you can configure the analysis to find two breakpoints. This is useful for more complex systems where processes might change across multiple frequency domains (e.g., long-term, seasonal, and short-term event scales).

## Forcing a Two-Breakpoint Analysis

To force the model to find two breakpoints, you need to pass a dictionary to the `segmented_opts` argument in the `run_analysis` method of the `Analysis` class. The key is `n_breakpoints` and the value should be `2`.

```python
from waterSpec.analysis import Analysis

# Assuming 'analysis' is an instance of the Analysis class
results = analysis.run_full_analysis(
    output_dir='my_output',
    # ... other options
    segmented_opts={'n_breakpoints': 2} # This forces a 2-breakpoint fit
)
```
*Note: The automatic model selection (`analysis_type='auto'`) will still compare the 2-breakpoint model's BIC to the standard model's BIC and choose the better one. The `segmented_opts` ensures that if a segmented model is considered, it will be one with two breakpoints.*

## Example: Generating and Analyzing a 2-Breakpoint Signal

Here is a full example of how to generate a synthetic signal with two known breakpoints and then run the analysis on it. This code is available in `examples/run_two_breakpoints_example.py`.

### Full Code

```python
import numpy as np
import pandas as pd
import os
from waterSpec.fitter import fit_segmented_spectrum
from waterSpec.plotting import plot_spectrum
from waterSpec.interpreter import get_scientific_interpretation, get_persistence_traffic_light, _format_period

# Copied and modified from waterSpec.interpreter
def interpret_results_2bp(fit_results, param_name="Parameter"):
    """
    Generates a human-readable interpretation for a 2-breakpoint fit.
    """
    n_breakpoints = fit_results.get('n_breakpoints')

    if n_breakpoints == 2:
        beta1, beta2, beta3 = fit_results['beta1'], fit_results['beta2'], fit_results['beta3']
        bp1, bp2 = fit_results['breakpoint1'], fit_results['breakpoint2']
        interp1, interp2, interp3 = get_scientific_interpretation(beta1), get_scientific_interpretation(beta2), get_scientific_interpretation(beta3)
        summary_text = (
            f"Segmented Analysis (2 Breakpoints) for: {param_name}\n"
            f"Breakpoint 1 Period ≈ {_format_period(bp1)}\n"
            f"Breakpoint 2 Period ≈ {_format_period(bp2)}\n"
            f"-----------------------------------\n"
            f"Segment 1 (Low-Frequency):\n"
            f"  β1 = {beta1:.2f} | {get_persistence_traffic_light(beta1)}\n"
            f"  Interpretation: {interp1}\n"
            f"-----------------------------------\n"
            f"Segment 2 (Mid-Frequency):\n"
            f"  β2 = {beta2:.2f} | {get_persistence_traffic_light(beta2)}\n"
            f"  Interpretation: {interp2}\n"
            f"-----------------------------------\n"
            f"Segment 3 (High-Frequency):\n"
            f"  β3 = {beta3:.2f} | {get_persistence_traffic_light(beta3)}\n"
            f"  Interpretation: {interp3}"
        )
        results_dict = {"analysis_type": "segmented", "n_breakpoints": 2, "beta1": beta1, "beta2": beta2, "beta3": beta3, "breakpoint1": bp1, "breakpoint2": bp2}
    else:
        # Fallback for other cases - this part is simplified as we are only expecting 2 breakpoints
        summary_text = "This interpreter is only for 2-breakpoint models."
        results_dict = {}

    results_dict["summary_text"] = summary_text
    return results_dict


def generate_segmented_data_two_breakpoints(n_points=200, slope1=-1.5, slope2=-0.5, slope3=-2.0, noise_std=0.1):
    """Generates synthetic data with two breakpoints for spectral analysis."""
    log_freq = np.linspace(np.log(0.01), np.log(10), n_points)
    log_power = np.zeros(n_points)

    bp1 = np.log(1)
    bp2 = np.log(5)

    idx1 = np.argmin(np.abs(log_freq - bp1))
    idx2 = np.argmin(np.abs(log_freq - bp2))

    log_power[:idx1] = slope1 * (log_freq[:idx1] - log_freq[0]) + 5
    log_power[idx1:idx2] = log_power[idx1-1] + slope2 * (log_freq[idx1:idx2] - log_freq[idx1-1])
    log_power[idx2:] = log_power[idx2-1] + slope3 * (log_freq[idx2:] - log_freq[idx2-1])

    log_power += np.random.normal(0, noise_std, n_points)

    return np.exp(log_freq), np.exp(log_power), "Two Breakpoint Signal"


if __name__ == '__main__':
    frequency, power, param_name = generate_segmented_data_two_breakpoints()
    output_dir = 'two_breakpoint_output'
    os.makedirs(output_dir, exist_ok=True)

    print("Running 2-breakpoint segmented analysis...")

    fit_results = fit_segmented_spectrum(frequency, power, n_breakpoints=2)

    interp_results = interpret_results_2bp(fit_results, param_name=param_name)

    results = {**fit_results, **interp_results}

    plot_path = os.path.join(output_dir, f"{param_name}_spectrum_plot.png")
    plot_spectrum(
        frequency,
        power,
        fit_results=results,
        analysis_type='segmented',
        output_path=plot_path,
        param_name=param_name
    )

    summary_path = os.path.join(output_dir, f"{param_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(results['summary_text'])

    print("Analysis complete.")
    print(f"Summary saved to: {summary_path}")
    print(f"Plot saved to: {plot_path}")

    if results.get('n_breakpoints') == 2:
        print(f"  Breakpoint 1 Freq: {results.get('breakpoint1'):.2f}")
        print(f"  Breakpoint 2 Freq: {results.get('breakpoint2'):.2f}")
        print(f"  Beta 1: {results.get('beta1'):.2f}")
        print(f"  Beta 2: {results.get('beta2'):.2f}")
        print(f"  Beta 3: {results.get('beta3'):.2f}")
```

### Output Figure

When you run the script, it will generate the following plot, which clearly shows the two breakpoints and the three different spectral slopes (β values).

![Two Breakpoint Fit](../../two_breakpoint_output/Two%20Breakpoint%20Signal_spectrum_plot.png)

### Output Summary

The script also produces a text file with a detailed summary of the findings:

```
Segmented Analysis (2 Breakpoints) for: Two Breakpoint Signal
Breakpoint 1 Period ≈ 1.1 days
Breakpoint 2 Period ≈ 0.2 days
-----------------------------------
Segment 1 (Low-Frequency):
  β1 = 1.50 | 🟢 Persistent / Subsurface Dominated (High Persistence)
  Interpretation: 1 < β < 3 (fBm-like): Strong persistence, suggesting transport is damped by storage.
-----------------------------------
Segment 2 (Mid-Frequency):
  β2 = 0.55 | 🟡 Mixed / Weak Persistence
  Interpretation: -0.5 < β < 1 (fGn-like): Weak persistence or anti-persistence, suggesting event-driven transport.
-----------------------------------
Segment 3 (High-Frequency):
  β3 = 2.10 | 🟢 Persistent / Subsurface Dominated (High Persistence)
  Interpretation: 1 < β < 3 (fBm-like): Strong persistence, suggesting transport is damped by storage.
```
*Note: The exact values may vary slightly due to the random noise added to the data.*

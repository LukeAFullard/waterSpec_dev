import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(frequency, power, fit_results=None, output_path=None, show=True):
    """
    Plots the power spectrum on a log-log scale and optionally shows the fitted line.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        fit_results (dict, optional): A dictionary with fitting results ('beta', 'intercept').
                                      If provided, the fitted line will be plotted. Defaults to None.
        output_path (str, optional): The path to save the plot image. If None, the plot
                                     is displayed interactively. Defaults to None.
        show (bool, optional): Whether to display the plot with plt.show().
                               Defaults to True.
    """
    plt.figure(figsize=(8, 6))

    # Plot the power spectrum
    plt.loglog(frequency, power, 'o', markersize=5, label='Power Spectrum')

    # Plot the fitted line if results are provided
    if fit_results:
        # Check for segmented fit results first
        beta1 = fit_results.get('beta1')
        beta2 = fit_results.get('beta2')
        breakpoint_freq = fit_results.get('breakpoint')

        if beta1 is not None and beta2 is not None and breakpoint_freq is not None:
            # Handle segmented plot
            plt.axvline(breakpoint_freq, color='k', linestyle='--', label=f'Breakpoint = {breakpoint_freq:.2f}')

            # Find the power value at the breakpoint to connect the lines
            # This requires an estimate of the intercept of the first line.
            # We can get this from the model summary if available, but it's complex.
            # A simpler way is to find the power of the data point closest to the breakpoint.
            idx = np.argmin(np.abs(frequency - breakpoint_freq))
            breakpoint_power = power[idx]

            # Line 1: P = C1 * f^-B1  => C1 = P_break * f_break^B1
            const1 = breakpoint_power * (breakpoint_freq ** beta1)
            freq1 = frequency[frequency < breakpoint_freq]
            fit_line1 = const1 * (freq1 ** -beta1)
            plt.loglog(freq1, fit_line1, 'r-', linewidth=2, label=f'Fit (β₁ = {beta1:.2f})')

            # Line 2: P = C2 * f^-B2 => C2 = P_break * f_break^B2
            const2 = breakpoint_power * (breakpoint_freq ** beta2)
            freq2 = frequency[frequency >= breakpoint_freq]
            fit_line2 = const2 * (freq2 ** -beta2)
            plt.loglog(freq2, fit_line2, 'g-', linewidth=2, label=f'Fit (β₂ = {beta2:.2f})')

        else:
            # Handle standard single-slope plot
            beta = fit_results.get('beta')
            intercept = fit_results.get('intercept')
            if beta is not None and intercept is not None:
                # Recreate the fitted line: log(power) = intercept - beta * log(frequency)
                # power = exp(intercept) * frequency ** -beta
                fit_line = np.exp(intercept) * (frequency ** -beta)
                plt.loglog(frequency, fit_line, 'r-', linewidth=2, label=f'Fit (β = {beta:.2f})')

    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    # Close the plot to free up memory, important for running many tests
    plt.close()

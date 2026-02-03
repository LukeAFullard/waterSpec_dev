
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, Union

def fit_carma_drw(
    time: np.ndarray,
    data: np.ndarray,
    errors: Optional[np.ndarray] = None
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Fits a Damped Random Walk (DRW), also known as a CARMA(1,0) or Ornstein-Uhlenbeck process,
    to irregularly sampled time series data.

    This model describes a process that behaves like a random walk at short timescales
    (dt << tau) and white noise at long timescales (dt >> tau).

    The process is defined by:
        dx(t) = - (1/tau) * (x(t) - mu) dt + sigma * dW(t)

    Args:
        time (np.ndarray): Time array (irregularly sampled).
        data (np.ndarray): Data values.
        errors (np.ndarray, optional): Measurement errors (Gaussian noise sigma).

    Returns:
        Dict:
            - "tau": Damping timescale (time units).
            - "sigma": Driving noise amplitude (units/sqrt(time)).
            - "mu": Mean level.
            - "psd_func": A callable function f(freqs) -> Power.
            - "beta_func": A callable function f(freqs) -> Spectral slope.
            - "nll": Negative log-likelihood.
            - "success": Boolean.
    """
    # 1. Sort Data
    sort_idx = np.argsort(time)
    t = time[sort_idx]
    y = data[sort_idx]
    dy = errors[sort_idx] if errors is not None else np.zeros_like(y)

    dt = np.diff(t)
    # Check for duplicate times
    if np.any(dt <= 0):
        raise ValueError("Time array must be strictly increasing.")

    # 2. Define Negative Log-Likelihood (NLL)
    # Using the exact transition density for OU process:
    # x(t+dt) | x(t) ~ Normal( mean_cond, var_cond )
    # mean_cond = mu + (x(t) - mu) * exp(-dt/tau)
    # var_cond = (sigma^2 * tau / 2) * (1 - exp(-2*dt/tau))
    # We also add measurement error variance: var_obs = var_cond + dy[i+1]^2

    def nll(params):
        # Unpack: log_tau, log_sigma, mu
        # Using log parameters ensures positivity
        tau = np.exp(params[0])
        sigma = np.exp(params[1])
        mu = params[2]

        # Determine conditional moments
        exp_factor = np.exp(-dt / tau)

        # Prediction for next step
        y_pred = mu + (y[:-1] - mu) * exp_factor

        # Variance of the process evolution
        # Asymptotic variance is sigma^2 * tau / 2
        var_proc = (sigma**2 * tau / 2.0) * (1.0 - np.exp(-2.0 * dt / tau))

        # Total variance including measurement error at step i+1
        var_total = var_proc + dy[1:]**2

        # Residuals
        resid = y[1:] - y_pred

        # Log Likelihood Sum
        # -0.5 * sum( log(2*pi*var) + resid^2 / var )
        # NLL = 0.5 * sum ...
        term1 = np.log(var_total)
        term2 = resid**2 / var_total

        return 0.5 * np.sum(term1 + term2)

    # 3. Optimize
    # Initial guesses
    # tau ~ total duration / 4? Or median dt?
    # sigma ~ std(diff(y)) / sqrt(median(dt))
    tau_init = (t[-1] - t[0]) / 10.0
    sigma_init = np.std(np.diff(y)) / np.sqrt(np.median(dt))
    mu_init = np.mean(y)

    initial_params = [np.log(tau_init), np.log(sigma_init), mu_init]

    res = minimize(nll, initial_params, method='L-BFGS-B')

    tau_fit = np.exp(res.x[0])
    sigma_fit = np.exp(res.x[1])
    mu_fit = res.x[2]

    # 4. Construct helper functions
    def psd_analytic(f):
        # P(f) = (2 * sigma^2 * tau^2) / (1 + (2*pi*f*tau)^2)
        # Note: definition of PSD scaling varies.
        # This is the Lorentizan form.
        omega = 2 * np.pi * f
        return (2 * sigma_fit**2 * tau_fit**2) / (1 + (omega * tau_fit)**2)

    def beta_analytic(f):
        # Beta is negative slope of log P vs log f
        # P ~ 1 / (1 + x^2)
        # log P ~ - log(1 + x^2)
        # d/d(log f) = f * d/df
        # dP/df = ...
        # beta = (2 * (2*pi*f*tau)^2) / (1 + (2*pi*f*tau)^2)
        # At low freq, beta -> 0. At high freq, beta -> 2.
        omega = 2 * np.pi * f
        x2 = (omega * tau_fit)**2
        return (2 * x2) / (1 + x2)

    return {
        "tau": tau_fit,
        "sigma": sigma_fit,
        "mu": mu_fit,
        "nll": res.fun,
        "success": res.success,
        "psd_func": psd_analytic,
        "beta_func": beta_analytic,
        "message": res.message
    }

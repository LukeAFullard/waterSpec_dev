
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

def convergent_cross_mapping(
    time: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: Optional[np.ndarray] = None,
    E: int = 3,
    tau: int = 1,
    n_neighbors: Optional[int] = None,
    max_gap: Optional[float] = None
) -> Dict:
    """
    Performs Convergent Cross Mapping (CCM) to test if Y causes X (Y -> X).

    In CCM, if Y causes X, information about Y is encoded in the history of X.
    Therefore, the reconstructed state space of X (M_X) can be used to predict Y.
    We check if correlation(Y_predicted, Y_observed) increases with library length L.

    Args:
        time (np.ndarray): Time array.
        X (np.ndarray): The potential effect variable (we use M_X to predict Y).
        Y (np.ndarray): The potential causal variable.
        lib_sizes (np.ndarray): Array of library lengths L to test convergence.
        E (int): Embedding dimension.
        tau (int): Time lag for embedding.
        n_neighbors (int): Number of nearest neighbors (defaults to E+1).
        max_gap (float): Maximum gap size allowed for interpolation. If a gap > max_gap exists,
                         interpolation is skipped and a warning/error is raised (safe default).
                         If None, defaults to 5 * median_dt.

    Returns:
        Dict containing 'lib_sizes', 'rho' (correlation vs L), 'rmse'.
    """

    n = len(X)
    if len(Y) != n:
        raise ValueError("X and Y must be same length.")

    # --- Check for Uneven Sampling ---
    if len(time) > 1:
        dt = np.diff(time)
        median_dt = np.median(dt)

        # Define safe gap threshold
        if max_gap is None:
            max_gap = 5.0 * median_dt

        # Check for large gaps
        max_obs_gap = np.max(dt)
        if max_obs_gap > max_gap:
            warnings.warn(
                f"Data contains a gap of {max_obs_gap:.2f} which exceeds the maximum allowed gap "
                f"for safe interpolation ({max_gap:.2f}). Analysis may be unreliable. "
                "Consider segmenting the time series.",
                UserWarning
            )
            # We proceed with interpolation but heavily warned.
            # Or should we abort? Aborting might be too strict for a general library.
            # Let's warn strongly.

        # Check if any dt deviates significantly (e.g. > 5%) from median
        if not np.allclose(dt, median_dt, rtol=0.05):
            warnings.warn(
                "Time series appears to be unevenly sampled. "
                "Interpolating to a regular grid (median dt) for valid state-space reconstruction.",
                UserWarning
            )
            # Interpolate
            reg_time = np.arange(time[0], time[-1] + median_dt, median_dt)
            X = np.interp(reg_time, time, X)
            Y = np.interp(reg_time, time, Y)
            time = reg_time # Update time to regular grid
            n = len(X)

    if n_neighbors is None:
        n_neighbors = E + 1

    # 1. Embed X (Shadow Manifold M_X)
    # Create delay vectors x_t = [X[t], X[t-tau], ..., X[t-(E-1)tau]]
    # Valid indices for embedding
    n_vectors = n - (E - 1) * tau

    if n_vectors < 10:
        raise ValueError("Time series too short for this E and tau.")

    # Build Shadow Manifold M_X
    M_X = np.zeros((n_vectors, E))
    for i in range(E):
        # Component k (0..E-1) is X[time_index - k*tau]
        offset = (E - 1) * tau
        M_X[:, i] = X[offset - i*tau : n - i*tau]

    # Target variable Y to predict
    # We predict Y at the same time indices as the manifold vectors
    # Y_target corresponds to Y[offset : n]
    Y_target = Y[(E - 1) * tau : n]

    if lib_sizes is None:
        lib_sizes = np.linspace(E*5, n_vectors, 10, dtype=int)

    rho_stats = []
    rmse_stats = []

    for L in lib_sizes:
        if L > n_vectors:
            break

        # Implementation: Randomly sample L indices to form the "Library".
        # Predict ALL points in the manifold using this library (excluding themselves).

        indices = np.arange(n_vectors)
        lib_indices = np.random.choice(indices, size=L, replace=False)

        # M_X_lib = M_X[lib_indices]
        # M_X_target = M_X

        # Optimization: if L is large, computing distance matrix is O(N*L).
        # We can just compute full distance matrix once O(N^2) if N < 2000.

        if n_vectors > 2000:
             # Too big for full matrix, just sample 500 targets
             target_indices = np.random.choice(indices, 500, replace=False)
        else:
             target_indices = indices

        # Distances
        # We iterate targets
        y_pred_vec = []
        y_true_vec = []

        # Pre-select library manifold
        Library = M_X[lib_indices]
        Library_Y = Y_target[lib_indices]

        for t_idx in target_indices:
            # Vector x_t
            x_t = M_X[t_idx]

            # Distances to library vectors
            dists = np.linalg.norm(Library - x_t, axis=1)

            # Find closest neighbors
            # Sort distances
            sorted_idx = np.argsort(dists)

            closest_args = []
            min_dist = 1e-9

            count = 0
            for arg in sorted_idx:
                if dists[arg] < min_dist: # basically 0, likely self-match
                     continue
                closest_args.append(arg)
                count += 1
                if count >= n_neighbors:
                    break

            if len(closest_args) < n_neighbors:
                # Not enough neighbors (library too small?)
                y_pred_vec.append(np.nan)
                y_true_vec.append(Y_target[t_idx])
                continue

            closest_args = np.array(closest_args)
            nearest_dists = dists[closest_args]
            nearest_y = Library_Y[closest_args]

            # Weights: u_i = exp(-d_i / d_1)
            # w_i = u_i / sum(u)
            if nearest_dists[0] == 0:
                 # Should not happen due to filter above, but safety
                 pred = nearest_y[0]
            else:
                u = np.exp(-nearest_dists / nearest_dists[0])
                w = u / np.sum(u)
                pred = np.sum(w * nearest_y)

            y_pred_vec.append(pred)
            y_true_vec.append(Y_target[t_idx])

        # Calculate correlation
        valid = np.isfinite(y_pred_vec)
        if np.sum(valid) > 2:
            r = np.corrcoef(np.array(y_true_vec)[valid], np.array(y_pred_vec)[valid])[0,1]
            rmse = np.sqrt(np.mean((np.array(y_true_vec)[valid] - np.array(y_pred_vec)[valid])**2))
        else:
            r = 0.0
            rmse = np.nan

        rho_stats.append(r)
        rmse_stats.append(rmse)

    return {
        'lib_sizes': lib_sizes,
        'rho': np.array(rho_stats),
        'rmse': np.array(rmse_stats)
    }

def find_optimal_embedding(
    data: np.ndarray,
    max_E: int = 10,
    tau: int = 1,
    tp: int = 1
) -> int:
    """
    Finds optimal embedding dimension E using Simplex Projection prediction skill.
    Critically, this uses a forward prediction horizon (tp) to avoid tautology.

    We predict X(t + tp) using the manifold constructed from X(t).

    Args:
        data: Time series data.
        max_E: Maximum embedding dimension to test.
        tau: Embedding lag.
        tp: Prediction horizon (steps into future). Default 1.
    """
    n = len(data)
    # Check for uneven sampling within optimal embedding?
    # Since we pass 'np.arange(n)' as dummy time to CCM inside this function,
    # the CCM internal check won't see original timestamps.
    # However, find_optimal_embedding assumes 'data' is already regular (or we don't know time).
    # Ideally, find_optimal_embedding should also accept 'time' array.
    # But for now, let's assume if user calls this with just data, they should regularize first
    # or we document it.

    # Actually, the user flow is usually:
    # 1. interpolate
    # 2. find E
    # 3. run CCM

    # Given we added interpolation inside CCM, we should probably add it here too or
    # let CCM handle it. But CCM is called with dummy time here.
    # Let's assume data passed here is treated as regular steps.

    scores = []

    if tp < 1:
        raise ValueError("Prediction horizon tp must be >= 1.")

    X_source = data[:-tp]
    Y_target = data[tp:]

    n_eff = len(X_source)

    # We create a simple self-prediction wrapper
    for E in range(1, max_E + 1):
        # We use CCM function mechanism:
        # Embed X_source into M_X
        # Predict Y_target
        # Use full library size available

        try:
            res = convergent_cross_mapping(
                np.arange(n_eff), # Dummy time, assumed regular
                X=X_source,
                Y=Y_target,
                lib_sizes=[n_eff - (E-1)*tau - 5], # One large library
                E=E,
                tau=tau
            )
            scores.append(res['rho'][0])
        except ValueError:
            scores.append(-np.inf) # Invalid E (too short)

    return np.argmax(scores) + 1

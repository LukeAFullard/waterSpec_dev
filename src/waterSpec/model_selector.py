import logging
import numpy as np

from .fitter import fit_standard_model, fit_segmented_spectrum

class ModelSelector:
    """
    Encapsulates the logic for selecting the best spectral model (standard vs segmented)
    based on BIC.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def select_best_model(
        self,
        frequency,
        power,
        fit_method,
        ci_method,
        bootstrap_type,
        n_bootstraps,
        p_threshold,
        max_breakpoints,
        seed,
    ):
        """
        Performs fits for models with different numbers of breakpoints and
        selects the best one using BIC.
        """
        self.logger.info("Performing model selection based on BIC...")
        all_models = []
        failed_model_reasons = []

        # Create a master SeedSequence from `seed`, then spawn child seeds.
        # We use SeedSequence.spawn(...) so each child seed is independent.
        if seed is not None:
            # np.random.SeedSequence accepts None or an int; passing an int ensures reproducibility.
            master_ss = np.random.SeedSequence(seed)
            child_seeds = master_ss.spawn(max_breakpoints + 1)
        else:
            child_seeds = [None] * (max_breakpoints + 1)

        # Fit the standard model (0 breakpoints)
        self.logger.info("Fitting standard model (0 breakpoints)...")
        try:
            standard_results = fit_standard_model(
                frequency,
                power,
                method=fit_method,
                ci_method=ci_method,
                bootstrap_type=bootstrap_type,
                n_bootstraps=n_bootstraps,
                seed=child_seeds[0],
                logger=self.logger,
            )
            if np.isfinite(standard_results.get("bic", np.inf)):
                standard_results["model_type"] = "standard"
                standard_results["n_breakpoints"] = 0
                all_models.append(standard_results)
                self.logger.info(
                    "Standard model fit complete. "
                    f"BIC: {standard_results['bic']:.2f}"
                )
            else:
                reason = standard_results.get("failure_reason", "Unknown error")
                failed_model_reasons.append(f"Standard model (0 breakpoints): {reason}")
                self.logger.warning(f"Standard model fit failed: {reason}")
        except (ValueError, ImportError) as e:
            reason = f"A critical error occurred during standard model setup: {e!r}"
            failed_model_reasons.append(f"Standard model (0 breakpoints): {reason}")
            self.logger.error(
                "Standard model fit crashed due to a critical error: %s", e, exc_info=True
            )
        except Exception as e:
            reason = f"An unexpected error occurred: {e!r}"
            failed_model_reasons.append(f"Standard model (0 breakpoints): {reason}")
            self.logger.error("Standard model fit crashed: %s", e, exc_info=True)

        # Fit segmented models (1 and possibly 2 breakpoints)
        for n_breakpoints in range(1, max_breakpoints + 1):
            self.logger.info(f"Fitting segmented model with {n_breakpoints} breakpoint(s)...")
            try:
                # Spawn a new seed for each model to ensure independent bootstrap samples.
                model_seed = child_seeds[n_breakpoints]
                seg_results = fit_segmented_spectrum(
                    frequency,
                    power,
                    n_breakpoints=n_breakpoints,
                    p_threshold=p_threshold,
                    ci_method=ci_method,
                    bootstrap_type=bootstrap_type,
                    n_bootstraps=n_bootstraps,
                    seed=model_seed,
                    logger=self.logger,
                )

                if np.isfinite(seg_results.get("bic", np.inf)):
                    seg_results["model_type"] = f"segmented_{n_breakpoints}bp"
                    all_models.append(seg_results)
                    self.logger.info(
                        f"Segmented model ({n_breakpoints} breakpoint(s)) fit complete. "
                        f"BIC: {seg_results['bic']:.2f}"
                    )
                else:
                    reason = seg_results.get("failure_reason", "Model did not converge or was not significant")
                    failed_model_reasons.append(
                        f"Segmented model ({n_breakpoints} breakpoint(s)): {reason}"
                    )
                    self.logger.warning(
                        f"Segmented model ({n_breakpoints} breakpoint(s)) fit failed: {reason}"
                    )
            except (ValueError, ImportError) as e:
                reason = f"A critical error occurred during segmented model setup: {e!r}"
                failed_model_reasons.append(
                    f"Segmented model ({n_breakpoints} breakpoint(s)): {reason}"
                )
                self.logger.error(
                    "Segmented model (%d breakpoint(s)) fit crashed due to a critical error: %s",
                    n_breakpoints,
                    e,
                    exc_info=True,
                )
            except Exception as e:
                reason = f"An unexpected error occurred: {e!r}"
                failed_model_reasons.append(
                    f"Segmented model ({n_breakpoints} breakpoint(s)): {reason}"
                )
                self.logger.error(
                    "Segmented model (%d breakpoint(s)) fit crashed: %s",
                    n_breakpoints,
                    e,
                    exc_info=True,
                )

        # Check if any model produced a valid (finite) BIC.
        valid_models = [m for m in all_models if np.isfinite(m.get("bic", np.inf))]

        if not valid_models:
            failure_summary = "All models failed; no valid BIC values found."
            if failed_model_reasons:
                failure_summary += " Reasons:\n" + "\n".join(
                    f"- {reason}" for reason in failed_model_reasons
                )
            self.logger.error(failure_summary)
            raise RuntimeError(failure_summary)

        best_model = min(valid_models, key=lambda x: x["bic"])
        self.logger.info(
            f"Best model selected: {best_model['model_type']} "
            f"(BIC: {best_model['bic']:.2f})"
        )

        fit_results = best_model.copy()
        fit_results["all_models"] = all_models
        fit_results["failed_model_reasons"] = failed_model_reasons
        fit_results["chosen_model"] = best_model["model_type"]
        fit_results["analysis_mode"] = "auto"
        fit_results["ci_method"] = ci_method

        if best_model["n_breakpoints"] == 0:
            fit_results["chosen_model_type"] = "standard"
        else:
            fit_results["chosen_model_type"] = "segmented"

        return fit_results

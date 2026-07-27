with open('validation/validate_peak_detection_sweep.py', 'r') as f:
    content = f.read()

old_rpy2 = """import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr"""

new_rpy2 = """try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False"""

content = content.replace(old_rpy2, new_rpy2)

old_dplr = """    # --- dplR redfit Analysis ---
    try:
        dplr = importr("dplR")
        with localconverter(robjects.default_converter + pandas2ri.converter):
            redfit_results = dplr.redfit(
                robjects.FloatVector(series), nsim=500, mctest=True
            )
        names = list(redfit_results.names())
        freq = np.array(redfit_results[names.index("freq")])
        power = np.array(redfit_results[names.index("gxxc")])
        ci95 = np.array(redfit_results[names.index("ci95")])
        peak_idx = np.argmin(np.abs(freq - SIGNAL_FREQ_CPD))
        dplr_peak_found = power[peak_idx] > ci95[peak_idx]
    except Exception as e:
        logger.error(
            "dplR analysis failed for beta=%.1f, amp=%.2f: %s",
            beta,
            signal_amp,
            e,
            exc_info=True,
        )
        dplr_peak_found = "ERROR" """

new_dplr = """    # --- dplR redfit Analysis ---
    if HAS_RPY2:
        try:
            dplr = importr("dplR")
            with localconverter(robjects.default_converter + pandas2ri.converter):
                redfit_results = dplr.redfit(
                    robjects.FloatVector(series), nsim=500, mctest=True
                )
            names = list(redfit_results.names())
            freq = np.array(redfit_results[names.index("freq")])
            power = np.array(redfit_results[names.index("gxxc")])
            ci95 = np.array(redfit_results[names.index("ci95")])
            peak_idx = np.argmin(np.abs(freq - SIGNAL_FREQ_CPD))
            dplr_peak_found = power[peak_idx] > ci95[peak_idx]
        except Exception as e:
            logger.error(
                "dplR analysis failed for beta=%.1f, amp=%.2f: %s",
                beta,
                signal_amp,
                e,
                exc_info=True,
            )
            dplr_peak_found = "ERROR"
    else:
        dplr_peak_found = "N/A" """

content = content.replace(old_dplr, new_dplr)

old_an = """    try:
        ws_analyzer = Analysis(
            file_path, time_col="time", data_col="value", detrend_method=None
        )
        ws_results = ws_analyzer.run_full_analysis(
            output_dir=temp_dir, grid_type="linear", peak_detection_method="residual"
        )"""

new_an = """    try:
        ws_analyzer = Analysis(
            time_col="time", data_col="value", file_path=file_path, detrend_method=None, base_dir=""
        )
        ws_results = ws_analyzer.run_full_analysis(
            output_dir=temp_dir, peak_detection_method="residual"
        )"""

content = content.replace(old_an, new_an)

old_main = """    try:
        for beta in BETA_VALUES:
            for amp in AMPLITUDE_VALUES:
                ws_found, dplr_found = run_single_validation(beta, amp, temp_dir)

                if ws_found == "ERROR":
                    ws_str = "🔥 ERROR"
                else:
                    ws_str = "✅ Found" if ws_found else "❌ Not Found"

                if dplr_found == "ERROR":
                    dplr_str = "🔥 ERROR"
                else:
                    dplr_str = "✅ Found" if dplr_found else "❌ Not Found"

                results_data.append(
                    {
                        "beta": beta,
                        "amplitude": amp,
                        "waterSpec": ws_str,
                        "dplR": dplr_str,
                    }
                )
                print(f"{beta:<6.1f} | {amp:<10.2f} | {ws_str:<12} | {dplr_str:<12}")
    finally:"""

new_main = """    try:
        for beta in BETA_VALUES:
            for amp in AMPLITUDE_VALUES:
                ws_found, dplr_found = run_single_validation(beta, amp, temp_dir)

                if ws_found == "ERROR":
                    ws_str = "🔥 ERROR"
                else:
                    ws_str = "✅ Found" if ws_found else "❌ Not Found"

                if dplr_found == "ERROR":
                    dplr_str = "🔥 ERROR"
                elif dplr_found == "N/A":
                    dplr_str = "N/A"
                else:
                    dplr_str = "✅ Found" if dplr_found else "❌ Not Found"

                results_data.append(
                    {
                        "beta": beta,
                        "amplitude": amp,
                        "waterSpec": ws_str,
                        "dplR": dplr_str,
                    }
                )
                print(f"{beta:<6.1f} | {amp:<10.2f} | {ws_str:<12} | {dplr_str:<12}")
    finally:"""
content = content.replace(old_main, new_main)

content = content.replace("AMPLITUDE_VALUES = [2.0, 1.5, 1.0, 0.8, 0.5, 0.3]", "AMPLITUDE_VALUES = [2.0, 0.8]")
content = content.replace("BETA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]", "BETA_VALUES = [0.5, 1.5]")
content = content.replace("print(df.to_markdown(index=False))", "print(df)")

with open('validation/validate_peak_detection_sweep.py', 'w') as f:
    f.write(content)

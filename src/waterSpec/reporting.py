import csv
import json
import os


class ReportGenerator:
    """
    Generates reports in JSON, CSV, and HTML formats for waterSpec analysis results.
    """

    def __init__(self, results, metadata=None):
        """
        Initialize the ReportGenerator.

        Args:
            results (dict): The full results dictionary returned by a waterSpec Analysis.
            metadata (dict, optional): Contextual information about the dataset.
                                       e.g., {"site": "USGS-012345", "variable": "Discharge"}
        """
        self.results = results
        self.metadata = metadata or {}

    def to_json(self, output_path):
        """
        Save the full nested results to a JSON file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # We need to handle numpy arrays and floats in the results dictionary
        # A simple custom JSON encoder to handle basic numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np

                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()

                try:
                    import pandas as pd

                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient="records")
                    if isinstance(obj, pd.Series):
                        return obj.tolist()
                except ImportError:
                    pass

                # Safe fallback for non-serializable objects (like class instances)
                if hasattr(obj, "__dict__"):
                    return str(type(obj))
                return super().default(obj)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": self.metadata, "results": self.results},
                f,
                indent=4,
                cls=NpEncoder,
            )

    def to_csv(self, output_path):
        """
        Save key metrics to a tabular CSV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Extract basic metrics based on what's available
        row = {
            "Site": self.metadata.get("site", "Unknown"),
            "Variable": self.metadata.get("variable", "Unknown"),
        }

        # Check for Haar results
        if "haar_results" in self.results:
            haar = self.results["haar_results"]
            row["Haar_Beta"] = haar.get("beta", "")
            if "segmented_results" in haar and haar["segmented_results"] is not None:
                row["Haar_Breakpoints"] = len(
                    haar["segmented_results"].get("breakpoints", [])
                )

        # Check for LS results (may be nested under spectral_results or at root)
        spec = self.results.get("spectral_results", self.results)
        if "chosen_model" in spec:
            spec = spec.get(spec["chosen_model"], spec)

        if "beta" in spec or "betas" in spec:
            row["LS_Beta"] = spec.get(
                "beta", spec.get("betas", [""])[0] if "betas" in spec else ""
            )

        # Check for Bivariate/Hysteresis results
        hyst = self.results.get("hysteresis_results", self.results)
        if "area" in hyst:
            row["Hysteresis_Area"] = hyst.get("area", "")
            row["Hysteresis_Direction"] = hyst.get("direction", "")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

    def _prepare_html_context(self):
        """Prepare variables needed to render the HTML template."""
        context = {
            "metadata": self.metadata,
            "interpretation": None,
            "metrics": {},
            "plots": [],
            "plot_errors": [],
        }

        # 1. Interpretation
        interp = self.results.get("interpretation", self.results)
        if "summary_text" in interp:
            context["interpretation"] = {
                "summary_text": interp.get("summary_text", ""),
                "uncertainty_warnings": interp.get("uncertainty_warnings", []),
            }
        else:
            from waterSpec.interpreter import interpret_results

            spec = self.results.get("spectral_results", self.results)
            if "beta" in spec or "betas" in spec:
                auto_interp = interpret_results(
                    spec,
                    param_name=self.metadata.get("variable", "Parameter"),
                )
                context["interpretation"] = {
                    "summary_text": auto_interp.get("summary_text", ""),
                    "uncertainty_warnings": auto_interp.get("uncertainty_warnings", []),
                }

        # 2. Metrics
        if "haar_results" in self.results:
            haar = self.results["haar_results"]
            beta = haar.get("beta", "N/A")
            if isinstance(beta, float):
                beta = f"{beta:.2f}"
            context["metrics"]["haar_beta"] = beta

        spec = self.results.get("spectral_results", self.results)
        if "chosen_model" in spec:
            spec = spec.get(spec["chosen_model"], spec)

        if "beta" in spec or "betas" in spec:
            beta = spec.get(
                "beta", spec.get("betas", ["N/A"])[0] if "betas" in spec else "N/A"
            )
            if isinstance(beta, float):
                beta = f"{beta:.2f}"
            context["metrics"]["ls_beta"] = beta

        hyst = self.results.get("hysteresis_results", self.results)
        if "area" in hyst:
            area = hyst.get("area", "N/A")
            if isinstance(area, float):
                area = f"{area:.4f}"
            context["metrics"]["hysteresis_area"] = area
            context["metrics"]["hysteresis_direction"] = hyst.get("direction", "N/A")

        # 3. Plots
        raw_plots = []
        for k, v in self.results.items():
            if k.endswith("_plot_path") and isinstance(v, str) and os.path.exists(v):
                raw_plots.append(
                    (k.replace("_plot_path", "").replace("_", " ").title(), v)
                )

        if "haar_results" in self.results and isinstance(
            self.results["haar_results"], dict
        ):
            haar = self.results["haar_results"]
            for k, v in haar.items():
                if (
                    k.endswith("_plot_path")
                    and isinstance(v, str)
                    and os.path.exists(v)
                ):
                    raw_plots.append(
                        (k.replace("_plot_path", "").replace("_", " ").title(), v)
                    )

        if "plot_paths" in self.metadata and isinstance(
            self.metadata["plot_paths"], dict
        ):
            for title, path in self.metadata["plot_paths"].items():
                if os.path.exists(path):
                    raw_plots.append((title, path))

        import base64

        for title, path in raw_plots:
            try:
                with open(path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode("utf-8")
                ext = os.path.splitext(path)[1][1:].lower()
                if ext == "jpg":
                    ext = "jpeg"
                elif ext == "svg":
                    ext = "svg+xml"
                mime_type = (
                    f"image/{ext}"
                    if ext in ["png", "jpeg", "gif", "svg+xml"]
                    else "image/png"
                )

                context["plots"].append(
                    {"title": title, "mime_type": mime_type, "b64_string": b64_string}
                )
            except Exception as e:
                context["plot_errors"].append(f"Error loading plot {title}: {str(e)}")

        return context

    def to_html(self, output_path):
        """
        Generate a standalone HTML report with embedded base64 plots using Jinja2.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            from jinja2 import Environment, PackageLoader, select_autoescape
        except ImportError as e:
            raise ImportError(
                "The jinja2 package is required to generate HTML reports. "
                "Install it with `pip install jinja2`."
            ) from e

        env = Environment(
            loader=PackageLoader("waterSpec", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template("report_template.html")

        context = self._prepare_html_context()
        html_content = template.render(**context)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

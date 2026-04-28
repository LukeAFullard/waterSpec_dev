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
                import pandas as pd

                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="records")
                if isinstance(obj, pd.Series):
                    return obj.tolist()
                # Safe fallback for non-serializable objects (like class instances)
                if hasattr(obj, "__dict__"):
                    return str(type(obj))
                return super(NpEncoder, self).default(obj)

        with open(output_path, "w") as f:
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

        # Check for LS results
        if "spectral_results" in self.results:
            spec = self.results["spectral_results"]
            # Auto model might nest the chosen model
            if "chosen_model" in spec:
                chosen = spec.get(spec["chosen_model"], spec)
            else:
                chosen = spec

            row["LS_Beta"] = chosen.get(
                "beta", chosen.get("betas", [""])[0] if "betas" in chosen else ""
            )

        # Check for Bivariate/Hysteresis results
        if "hysteresis_results" in self.results:
            hyst = self.results["hysteresis_results"]
            row["Hysteresis_Area"] = hyst.get("area", "")
            row["Hysteresis_Direction"] = hyst.get("direction", "")

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

    def to_html(self, output_path):
        """
        Generate a standalone HTML report with embedded base64 plots.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>waterSpec Analysis Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }",
            "h1, h2, h3 { color: #0056b3; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".plot-container { margin: 20px 0; text-align: center; border: 1px solid #ccc; padding: 10px; background: #fafafa; }",
            "img { max-width: 100%; height: auto; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>waterSpec Analysis Report</h1>",
        ]

        # Metadata Section
        html_parts.append("<h2>Metadata</h2>")
        html_parts.append("<ul>")
        for k, v in self.metadata.items():
            html_parts.append(f"<li><strong>{k}:</strong> {v}</li>")
        html_parts.append("</ul>")

        # Interpretation Section
        if "interpretation" in self.results:
            html_parts.append("<h2>Interpretation Summary</h2>")
            html_parts.append(
                f"<pre style='background:#f4f4f4; padding:10px; border-left:4px solid #0056b3;'>{self.results['interpretation'].get('summary_text', '')}</pre>"
            )

            warnings = self.results["interpretation"].get("uncertainty_warnings", [])
            if warnings:
                html_parts.append(
                    "<div style='color:red; margin-top:10px;'><strong>Warnings:</strong><ul>"
                )
                for w in warnings:
                    html_parts.append(f"<li>{w}</li>")
                html_parts.append("</ul></div>")
        else:
            # We can use the interpreter directly if spectral_results are available but not yet interpreted
            from waterSpec.interpreter import interpret_results

            if "spectral_results" in self.results:
                interpretation = interpret_results(
                    self.results["spectral_results"],
                    param_name=self.metadata.get("variable", "Parameter"),
                )
                html_parts.append("<h2>Interpretation Summary</h2>")
                html_parts.append(
                    f"<pre style='background:#f4f4f4; padding:10px; border-left:4px solid #0056b3;'>{interpretation.get('summary_text', '')}</pre>"
                )
                warnings = interpretation.get("uncertainty_warnings", [])
                if warnings:
                    html_parts.append(
                        "<div style='color:red; margin-top:10px;'><strong>Warnings:</strong><ul>"
                    )
                    for w in warnings:
                        html_parts.append(f"<li>{w}</li>")
                    html_parts.append("</ul></div>")

        # Results Summary Table
        html_parts.append("<h2>Key Metrics</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")

        if "haar_results" in self.results:
            haar = self.results["haar_results"]
            beta = haar.get("beta", "N/A")
            if isinstance(beta, float):
                beta = f"{beta:.2f}"
            html_parts.append(f"<tr><td>Haar Beta</td><td>{beta}</td></tr>")

        if "spectral_results" in self.results:
            spec = self.results["spectral_results"]
            if "chosen_model" in spec:
                chosen = spec.get(spec["chosen_model"], spec)
            else:
                chosen = spec

            beta = chosen.get(
                "beta", chosen.get("betas", ["N/A"])[0] if "betas" in chosen else "N/A"
            )
            if isinstance(beta, float):
                beta = f"{beta:.2f}"
            html_parts.append(f"<tr><td>LS Beta</td><td>{beta}</td></tr>")

        if "hysteresis_results" in self.results:
            hyst = self.results["hysteresis_results"]
            area = hyst.get("area", "N/A")
            if isinstance(area, float):
                area = f"{area:.4f}"
            html_parts.append(f"<tr><td>Hysteresis Area</td><td>{area}</td></tr>")
            html_parts.append(
                f"<tr><td>Hysteresis Direction</td><td>{hyst.get('direction', 'N/A')}</td></tr>"
            )

        html_parts.append("</table>")

        html_parts.append("</body>")
        html_parts.append("</html>")

        with open(output_path, "w") as f:
            f.write("\n".join(html_parts))

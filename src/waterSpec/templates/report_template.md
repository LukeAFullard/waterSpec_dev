# waterSpec Analysis Report

{% if metadata %}
## Metadata

{% for k, v in metadata.items() %}
- **{{ k }}:** {{ v }}
{% endfor %}
{% endif %}

{% if data_characteristics %}
## Data Characteristics

{% for char in data_characteristics %}
- {{ char }}
{% endfor %}
{% endif %}

## Key Metrics

| Metric | Value |
| --- | --- |
{% if metrics.haar_beta %}| Haar Beta | {{ metrics.haar_beta }} |
{% endif %}{% if metrics.ls_beta %}| LS Beta | {{ metrics.ls_beta }} |
{% endif %}{% if metrics.hysteresis_area %}| Hysteresis Area | {{ metrics.hysteresis_area }} |
| Hysteresis Direction | {{ metrics.hysteresis_direction }} |
{% endif %}

{% if interpretation %}
## Spectral Fits

```
{{ interpretation.summary_text }}
```

{% if interpretation.uncertainty_warnings %}
## Bootstrapped Uncertainty

**Warnings:**
{% for w in interpretation.uncertainty_warnings %}
- {{ w }}
{% endfor %}
{% endif %}
{% endif %}

{% if methodological_caveats %}
## Methodological Caveats

{% for caveat in methodological_caveats %}
- {{ caveat }}
{% endfor %}
{% endif %}

{% if plots %}
## Visualizations

{% for plot in plots %}
### {{ plot.title }}

{% if plot.local_path %}
![{{ plot.title }} Plot]({{ plot.local_path }})
{% else %}
<img src="data:{{ plot.mime_type }};base64,{{ plot.b64_string }}" alt="{{ plot.title }} Plot" />
{% endif %}
{% endfor %}
{% endif %}

{% if plot_errors %}
## Errors

{% for error in plot_errors %}
- *{{ error }}*
{% endfor %}
{% endif %}

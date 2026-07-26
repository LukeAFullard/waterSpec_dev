# Section 12: Model Selection Logic (ModelSelector, AIC/BIC)

This report summarizes the findings from executing Section 12 of the `waterSpec` validation plan, testing model selection logic.

## 12.1 Standard-vs-segmented selection accuracy
- **Result**: FAIL
- **Details**: The pass rate was 0.00. The BIC checks failed to consistently select the correct model class. The segmented model's BIC did not robustly outperform the standard model on true breaks, or vice versa, possibly due to low statistical power or poor fitting at N=512 points.

## 12.2 Borderline cases
- **Result**: FAIL
- **Details**: The selection rates were erratic as slope differences increased ({0.2: 0.5, 0.5: 1.0, 1.0: 0.5, 2.0: 1.0}). The ModelSelector's preference for the segmented model did not increase monotonically with the strength of the true break, showing inconsistent borderline behavior.

## 12.3 LS-vs-Haar method agreement on model class
- **Result**: FAIL
- **Details**: Agreement rate was 0.00 at 30% missingness. The Lomb-Scargle and Haar analysis paths completely disagreed on model selection (standard vs segmented) when evaluated on identically subsampled true-break data.

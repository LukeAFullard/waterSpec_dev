# Convergent Cross Mapping (CCM) for Causality Detection

## Overview

Convergent Cross Mapping (CCM) is a method based on state-space reconstruction (Takens' Theorem) used to distinguish causality from correlation in non-linear dynamic systems. It was introduced by Sugihara et al. (Science, 2012).

### Why use CCM in Water Quality Analysis?

Environmental variables (Algae, Nutrients, Dissolved Oxygen) often exhibit non-linear dynamics where standard linear correlations fail.
*   **Non-linearity:** Two variables can be causally linked but have zero correlation (e.g., they are out of phase).
*   **Directionality:** Correlation is symmetric ($Corr(X,Y) = Corr(Y,X)$). Causality is not. CCM can determine if $X \to Y$, $Y \to X$, or bidirectional $X \leftrightarrow Y$.

## How it Works

If variable $Y$ causes variable $X$ ($Y \to X$), then information about the state of $Y$ is encoded in the history of $X$.
Therefore, we can reconstruct the "Shadow Manifold" $M_X$ from the time-lagged history of $X$, and use specific points on $M_X$ to predict the concurrent values of $Y$.

**The Key Signature of Causality:**
If $Y \to X$, the prediction skill (correlation between predicted and observed $Y$) **increases** (converges) as the library length $L$ (amount of data used) increases.

## Usage Example

```python
import numpy as np
from waterSpec.causality import convergent_cross_mapping, find_optimal_embedding

# 1. Load your data (must be regularly sampled)
# If irregular, interpolate first!
time = ...
Algae = ...
Phosphorus = ...

# 2. Determine Optimal Embedding Dimension (E)
E_opt = find_optimal_embedding(Algae, max_E=10)

# 3. Test: Does Phosphorus cause Algae? (P -> Algae)
# We use Algae manifold to predict Phosphorus
res_P_cause_Alg = convergent_cross_mapping(
    time,
    X=Algae,        # Effect (manifold source)
    Y=Phosphorus,   # Cause (target to predict)
    E=E_opt,
    tau=1
)

# 4. Test: Does Algae cause Phosphorus? (Algae -> P)
# We use Phosphorus manifold to predict Algae
res_Alg_cause_P = convergent_cross_mapping(
    time,
    X=Phosphorus,
    Y=Algae,
    E=E_opt,
    tau=1
)

# 5. Plot Convergence
import matplotlib.pyplot as plt
plt.plot(res_P_cause_Alg['lib_sizes'], res_P_cause_Alg['rho'], label='P causes Algae')
plt.plot(res_Alg_cause_P['lib_sizes'], res_Alg_cause_P['rho'], label='Algae causes P')
plt.xlabel('Library Length L')
plt.ylabel('Prediction Skill (rho)')
plt.legend()
plt.show()
```

### Interpretation
*   **Convergence:** If the curve rises and plateaus, causality is inferred.
*   **Dominant Direction:** The relationship with the higher $\rho$ at convergence is the stronger causal link. Note: Counter-intuitively, if interaction is strong, the "Effect" predicts the "Cause" better than vice-versa in CCM logic. So if `M_X` predicts `Y` well, `Y` causes `X`.

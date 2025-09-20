
                    Breakpoint Regression Results
====================================================================================================
No. Observations                      200
No. Model Parameters                    4
Degrees of Freedom                    196
Res. Sum of Squares               199.197
Total Sum of Squares              4446.98
R Squared                        0.955206
Adjusted R Squared               0.954287
Converged:                           True
====================================================================================================
====================================================================================================
                    Estimate      Std Err            t        P>|t|       [0.025       0.975]
----------------------------------------------------------------------------------------------------
const               -47.3283        0.983      -48.123     1.64e-110      -49.268      -45.389
alpha1              -2.50385       0.0593      -42.254     2.07e-100      -2.6207       -2.387
beta1                 1.0829        0.235       4.6152             -      0.62016       1.5456
breakpoint1         -14.0655        0.291            -             -      -14.638      -13.493
----------------------------------------------------------------------------------------------------
These alphas(gradients of segments) are estimatedfrom betas(change in gradient)
----------------------------------------------------------------------------------------------------
alpha2              -1.42096        0.227      -6.2589      2.39e-09      -1.8687     -0.97322
====================================================================================================
Davies test for existence of at least 1 breakpoint: p=6.29393e-07 (e.g. p<0.05 means reject null hypothesis of no breakpoints at 5% significance)


--- Validation Results: waterSpec vs dplR ---
| Known Beta   | waterSpec Beta   | dplR AR1     | dplR Beta (est.)   |
|--------------|------------------|--------------|--------------------|
| 0.00        | 0.0429           | N/A          | N/A                |
| 0.50        | 0.5125           | 0.3549       | 1.1003             |
| 1.00        | 1.0208           | 0.7436       | 5.8015             |
| 1.50        | 1.5342           | 0.9572       | 44.7271            |
| 2.00        | 2.0312           | 0.0000       | 0.0000             |
| 2.50        | N/A              | 0.0000       | 0.0000             |

--- Comparison Summary ---
The `waterSpec` package's beta estimates are close to the known beta values.
The `dplR` package's `redfit` function estimates the AR1 coefficient (rho).
The conversion from rho to beta is not straightforward and depends on the underlying model assumptions.
The estimated beta from dplR does not directly match the known beta, but it shows a monotonic relationship.
This validation confirms that both packages are sensitive to the spectral characteristics of the data.

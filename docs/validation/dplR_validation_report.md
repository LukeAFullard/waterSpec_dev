
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
| Known Beta   | waterSpec Beta   | dplR Beta    |
|--------------|------------------|--------------|
| 0.00        | 0.0429           | N/A          |
| 0.50        | 0.5125           | 0.6001       |
| 1.00        | 1.0208           | 1.0059       |
| 1.50        | 1.5342           | 1.5726       |
| 2.00        | 2.0312           | 2.0753       |
| 2.50        | N/A              | 2.5684       |

--- Comparison Summary ---
This validation compares the spectral exponent (beta) estimated by `waterSpec`
with a beta estimated from the bias-corrected spectrum returned by `dplR::redfit`.
The `dplR` beta is calculated by fitting a linear regression to the log-log spectrum
over a fixed frequency band (0.01 to 0.2).

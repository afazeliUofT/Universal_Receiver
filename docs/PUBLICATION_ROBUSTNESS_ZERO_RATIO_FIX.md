# Publication robustness zero-handling polish

This patch is a final publication-figure polish only.

It fixes two issues in the consolidated robustness summary:
1. when both UPAIR and the best classical receiver have BER/BLER equal to zero, the ratio is now recorded as `1.0` instead of `inf`
2. when a BER/BLER panel contains all-zero observed errors, the plot now shows the curves at a positive floor with an annotation instead of producing a blank log-scale panel

It also makes the BER/BLER gain figure explicitly mark capped points when UPAIR has zero observed error but the best classical receiver is still nonzero.

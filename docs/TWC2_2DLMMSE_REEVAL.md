This package does not retrain UPAIR.

It re-evaluates the existing three final mild checkpoints into new output folders:
- outputs/twc2_mild_main/
- outputs/twc2_mild_clean/
- outputs/twc2_mild_dmrsrich/

Main changes relative to the current add-2dlmmse path:
- removes both configured-reservoir curves from the new TWC2 campaign
- keeps perfect CSI as the exact-channel upper bound
- reevaluates LS+2D-LMMSE+LMMSE as a true frequency-time empirical LMMSE baseline
- disables spatial smoothing for this baseline
- uses a dedicated covariance cache name for the corrected baseline
- increases covariance-estimation strength and evaluation strength
- writes a new final figure set to TWC_plots2/
- hides BER/BLER points whenever the Monte Carlo evidence is insufficient
